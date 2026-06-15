#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

static std::string read_file(const std::string & path) {
    std::ifstream file(path);
    if (!file.good()) {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(1);
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static bool expect(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
    }
    return ok;
}

static std::string slice_between(const std::string & text, const std::string & begin, const std::string & end) {
    const size_t b = text.find(begin);
    if (b == std::string::npos) {
        return {};
    }
    const size_t e = text.find(end, b);
    if (e == std::string::npos) {
        return text.substr(b);
    }
    return text.substr(b, e - b);
}

int main(int argc, char ** argv) {
    bool ok = true;

    ok &= expect(argc == 2, "expected repo root argument");
    if (!ok) {
        return 1;
    }

    const std::string root = argv[1];
    const std::string vec = read_file(root + "/ggml/src/ggml-cuda/fattn-vec.cuh");
    const std::string helpers = slice_between(vec,
            "static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device()",
            "template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>");
    const std::string kernel = slice_between(vec,
            "template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>",
            "static __global__ void flash_attn_ext_vec");
    const std::string body = slice_between(vec,
            "static __global__ void flash_attn_ext_vec",
            "template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>");

    ok &= expect(!helpers.empty(),
        "VEC FA launch-bound helpers must be declared before the kernel template");
    ok &= expect(helpers.find("ggml_cuda_fattn_vec_is_turbo_kv_type") != std::string::npos,
        "VEC FA must centralize Turbo/TCQ type detection");
    ok &= expect(helpers.find("GGML_TYPE_TURBO2_0") != std::string::npos &&
                 helpers.find("GGML_TYPE_TURBO3_0") != std::string::npos &&
                 helpers.find("GGML_TYPE_TURBO4_0") != std::string::npos &&
                 helpers.find("GGML_TYPE_TURBO2_TCQ") != std::string::npos &&
                 helpers.find("GGML_TYPE_TURBO3_TCQ") != std::string::npos &&
                 helpers.find("GGML_TYPE_TURBO4_TCQ") != std::string::npos,
        "VEC FA Turbo/TCQ detection must cover all Turbo cache families");
    ok &= expect(helpers.find("ggml_cuda_fattn_vec_get_min_blocks") != std::string::npos,
        "VEC FA must compute launch-bound min blocks from K/V cache types");
    ok &= expect(helpers.find("? 2 : 1") != std::string::npos,
        "VEC FA min blocks must be 2 only for Turbo/TCQ pairs and 1 otherwise");

    ok &= expect(kernel.find("ggml_cuda_fattn_vec_get_min_blocks<type_K, type_V>()") != std::string::npos,
        "VEC FA launch_bounds must use the K/V type-aware min-block policy");
    ok &= expect(kernel.find("ggml_cuda_fattn_vec_get_nthreads_device(), 2") == std::string::npos,
        "VEC FA launch_bounds must not force all cache types to minBlocksPerSM=2");

    ok &= expect(body.find("constexpr bool K_is_turbo = ggml_cuda_fattn_vec_is_turbo_kv_type<type_K>();") != std::string::npos &&
                 body.find("constexpr bool V_is_turbo = ggml_cuda_fattn_vec_is_turbo_kv_type<type_V>();") != std::string::npos,
        "VEC FA kernel body must reuse the centralized Turbo/TCQ type helper");
    ok &= expect(body.find("sparse_v_threshold") == std::string::npos &&
                 body.find("if (dominated) { continue; }") == std::string::npos,
        "VEC FA standard V path must not use a data-dependent sparse V skip");

    return ok ? 0 : 1;
}
