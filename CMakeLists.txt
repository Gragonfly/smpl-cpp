cmake_minimum_required(VERSION 3.0.0)
project(mysmpl)

# json
find_package(nlohmann_json REQUIRED)

# eigen
find_package(Eigen3 REQUIRED)


file(GLOB HEADER
    "src/header/*.h"
)

file(GLOB SOURCE
    "src/*.cpp"
)

set(MYSMPL_INCLUDE_DIRS
    "src/header"
)

set(PROJECT_INCLUDE_DIRS
    ${MYSMPL_INCLUDE_DIRS}
)

set(PROJECT_LIBRARIES
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    stdc++fs
)

add_executable(mysmpl ${HEADER} ${SOURCE})
target_include_directories(mysmpl PRIVATE ${PROJECT_INCLUDE_DIRS})
target_link_libraries(mysmpl ${PROJECT_LIBRARIES})



# include_directories("C:\\Program Files (x86)\\Eigen3\\include")
# include_directories("C:\\Program Files (x86)\\nlohmann_json\\include")