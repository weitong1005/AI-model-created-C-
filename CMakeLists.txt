cmake_minimum_required(VERSION 3.10)
project(AI3700)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 确保中文正常处理
add_compile_options(-finput-charset=UTF-8)
include_directories(src)

add_executable(AI3700
        main.cpp
        src/file_utils.cpp
        src/file_utils.h
        src/hanzi_encoder.cpp
        src/hanzi_encoder.h
        src/matrix.cpp
        src/matrix.h
        src/rnn_model.cpp
        src/rnn_model.h
        src/start_model.cpp
        src/start_model.h
)

# 针对不同平台的特定设置
if(WIN32)
    # Windows 特定设置
    target_compile_definitions(AI3700 PRIVATE _CRT_SECURE_NO_WARNINGS)
elseif(UNIX)
    # Linux 特定设置
    target_compile_options(AI3700 PRIVATE -Wall -Wextra)
endif()

# 自动复制 hanzi.txt 到构建目录
add_custom_command(
        TARGET AI3700 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/data/hanzi.txt
        $<TARGET_FILE_DIR:AI3700>
)