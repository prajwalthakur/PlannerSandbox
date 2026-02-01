cmake_minimum_required(VERSION 3.8)
project(hybrid_rvo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(gazebo_msgs REQUIRED)

add_executable(hybrid_rvo src/hybrid_rvo.cpp)
target_include_directories(hybrid_rvo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(hybrid_rvo PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  hybrid_rvo
  "rclcpp"
  "std_msgs"
  "nav_msgs"
  "geometry_msgs"
  "yaml-cpp"
)

install(TARGETS hybrid_rvo
  DESTINATION lib/${PROJECT_NAME})



# spawn hol-robot  
add_executable(spawn_hol_robot src/spawn_hol_robot_node.cpp)
target_include_directories(spawn_hol_robot PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(spawn_hol_robot PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  spawn_hol_robot
  "rclcpp"
  "std_msgs"
  "nav_msgs"
  "geometry_msgs"
  "yaml-cpp"
  "gazebo_msgs"
)
target_link_libraries(spawn_hol_robot yaml-cpp)
install(TARGETS spawn_hol_robot
  DESTINATION lib/${PROJECT_NAME})



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
install(
  DIRECTORY robot_model
  DESTINATION share/${PROJECT_NAME}
)
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)
ament_package()
