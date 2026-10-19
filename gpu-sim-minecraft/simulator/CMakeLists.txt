cmake_minimum_required(VERSION 3.24)

project(minecraft_sim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

include(FetchContent)
FetchContent_Declare(fast_csv
    GIT_REPOSITORY https://github.com/ben-strasser/fast-cpp-csv-parser.git
    GIT_TAG 574a9fe4d323ba63416877a4a5fe59088d37aa34
)
FetchContent_MakeAvailable(fast_csv)

set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(minecraft_sim STATIC
    src/build_info.cpp src/math.cpp src/comparison.cpp src/recording.cpp
)
add_library(minecraft_sim::minecraft_sim ALIAS minecraft_sim)
target_include_directories(minecraft_sim PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_include_directories(minecraft_sim SYSTEM PRIVATE "${fast_csv_SOURCE_DIR}")
set_source_files_properties(src/recording.cpp PROPERTIES COMPILE_DEFINITIONS CSV_IO_NO_THREAD)
target_compile_features(minecraft_sim PUBLIC cxx_std_20)

add_executable(minecraft_sim_runner apps/main.cpp)
target_link_libraries(minecraft_sim_runner PRIVATE minecraft_sim::minecraft_sim nlohmann_json::nlohmann_json)

add_executable(minecraft_sim_compare apps/compare.cpp)
target_link_libraries(minecraft_sim_compare PRIVATE minecraft_sim::minecraft_sim)

add_executable(minecraft_sim_benchmark apps/benchmark.cpp)
target_link_libraries(minecraft_sim_benchmark PRIVATE minecraft_sim::minecraft_sim)
target_compile_definitions(minecraft_sim_benchmark PRIVATE
    MCSIM_BENCHMARK_ENVIRONMENT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/environments/benchmark.json"
)

foreach(target IN ITEMS minecraft_sim_runner minecraft_sim_compare minecraft_sim_benchmark)
    target_link_libraries(${target} PRIVATE nlohmann_json::nlohmann_json)
    target_compile_definitions(${target} PRIVATE
        MCSIM_DEFAULT_ENVIRONMENT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/environments/duel.json"
    )
endforeach()

set(simulator_targets minecraft_sim minecraft_sim_runner minecraft_sim_compare minecraft_sim_benchmark)
add_library(minecraft_sim_training SHARED apps/training.cpp)
target_link_libraries(minecraft_sim_training PRIVATE minecraft_sim::minecraft_sim nlohmann_json::nlohmann_json)
file(GLOB_RECURSE native_identity_files CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/apps/*.hpp"
)
list(APPEND native_identity_files "${CMAKE_CURRENT_SOURCE_DIR}/apps/training.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
list(SORT native_identity_files)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${native_identity_files})
set(native_identity "")
foreach(source IN LISTS native_identity_files)
    file(SHA256 "${source}" source_hash)
    file(RELATIVE_PATH relative_source "${CMAKE_CURRENT_SOURCE_DIR}" "${source}")
    string(APPEND native_identity "${relative_source}:${source_hash}\n")
endforeach()
string(SHA256 native_source_hash "${native_identity}")
set(native_revision "unknown")
find_package(Git QUIET)
if(Git_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE git_revision
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE git_result)
    if(git_result EQUAL 0)
        set(native_revision "${git_revision}")
    endif()
endif()
target_compile_definitions(minecraft_sim_training PRIVATE
    MCSIM_DEFAULT_ENVIRONMENT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/environments/duel.json"
    MCSIM_SOURCE_HASH="${native_source_hash}"
    MCSIM_SOURCE_REVISION="${native_revision}"
    MCSIM_BUILD_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
    MCSIM_BUILD_VERSION="${PROJECT_VERSION}"
)
set_target_properties(minecraft_sim PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(minecraft_sim_training PROPERTIES PREFIX "")
find_package(OpenMP QUIET COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(minecraft_sim_training PRIVATE OpenMP::OpenMP_CXX)
endif()
list(APPEND simulator_targets minecraft_sim_training)
if(BUILD_TESTING)
    add_executable(minecraft_sim_checks tests/core_checks.cpp)
    target_link_libraries(minecraft_sim_checks PRIVATE minecraft_sim::minecraft_sim nlohmann_json::nlohmann_json)
    target_compile_definitions(minecraft_sim_checks PRIVATE
        MCSIM_CONTRACT_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../contracts/policy_fixtures.json"
    )
    list(APPEND simulator_targets minecraft_sim_checks)
    add_test(NAME shared_core_checks COMMAND minecraft_sim_checks)
endif()

option(MCSIM_ENABLE_CUDA "Build the optional GPU execution backend" OFF)
if(MCSIM_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(minecraft_sim_cuda STATIC src/cuda_backend.cu)
    target_link_libraries(minecraft_sim_cuda PUBLIC minecraft_sim::minecraft_sim CUDA::cudart)
    set_target_properties(minecraft_sim_cuda PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(minecraft_sim_cuda PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false;--prec-div=true;--prec-sqrt=true>"
    )
    foreach(target IN ITEMS minecraft_sim_runner minecraft_sim_checks minecraft_sim_benchmark minecraft_sim_training)
        if(TARGET ${target})
            target_link_libraries(${target} PRIVATE minecraft_sim_cuda)
            target_compile_definitions(${target} PRIVATE MCSIM_HAS_CUDA=1)
        endif()
    endforeach()
    if(BUILD_TESTING)
        add_test(NAME shared_core_cuda_checks COMMAND minecraft_sim_checks --cuda)
        set_tests_properties(shared_core_cuda_checks PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

if(MINGW)
    get_filename_component(compiler_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    foreach(runtime IN ITEMS libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll libgomp-1.dll)
        set(runtime_path "${compiler_bin}/${runtime}")
        if(EXISTS "${runtime_path}")
            foreach(target IN LISTS simulator_targets)
                if(NOT target STREQUAL "minecraft_sim")
                    add_custom_command(TARGET ${target} POST_BUILD
                        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                            "${runtime_path}" "$<TARGET_FILE_DIR:${target}>"
                        VERBATIM
                    )
                endif()
            endforeach()
        endif()
    endforeach()
endif()

foreach(target IN LISTS simulator_targets)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /fp:strict)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off
        )
    endif()
endforeach()

if(BUILD_TESTING)
    add_test(NAME runner_duel COMMAND minecraft_sim_runner --duel --ticks 80)
    set_tests_properties(runner_duel PROPERTIES PASS_REGULAR_EXPRESSION "Duel ended: player [01] died")
    add_test(NAME benchmark_help COMMAND minecraft_sim_benchmark --help)
    set_tests_properties(benchmark_help PROPERTIES PASS_REGULAR_EXPRESSION "environment-ticks/sec")
    add_test(NAME runner_build_info COMMAND minecraft_sim_runner)
    set_tests_properties(runner_build_info PROPERTIES
        PASS_REGULAR_EXPRESSION "Target: Minecraft Java 26[.]3"
    )
endif()