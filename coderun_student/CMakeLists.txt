cmake_minimum_required(VERSION 3.8)
project(coderun_student)

find_package(ament_cmake REQUIRED)

# Python: with --symlink-install this is a link to python/solution.py, so edits apply without a rebuild
install(PROGRAMS python/solution.py DESTINATION lib/${PROJECT_NAME})

ament_package()
