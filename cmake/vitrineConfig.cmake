# CMake package config for vitrine.
#
#   find_package(vitrine CONFIG REQUIRED)
#   target_link_libraries(your_target PRIVATE vitrine::vitrine)
#
# The crate is built by cargo, not by CMake, so this file does not build anything: it locates the
# artefacts cargo produced and exposes them as one imported target. Install layout expected:
#
#   <prefix>/lib/cmake/vitrine/vitrineConfig.cmake      <- this file
#   <prefix>/lib/libvitrine.a   (or libvitrine.so)
#   <prefix>/include/vitrine.h
#
# A source-tree checkout works too: point CMAKE_PREFIX_PATH at the repository root, where
# target/release/libvitrine.a and include/vitrine.h already live after `cargo build --release`.

get_filename_component(VITRINE_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

if(NOT TARGET vitrine::vitrine)
    add_library(vitrine::vitrine STATIC IMPORTED)

    set(VITRINE_INCLUDE_DIR "${VITRINE_PREFIX}/include")
    set(VITRINE_STATIC "${VITRINE_PREFIX}/lib/libvitrine.a")
    set(VITRINE_SHARED "${VITRINE_PREFIX}/lib/libvitrine.so")

    if(EXISTS "${VITRINE_STATIC}")
        set(_vitrine_lib "${VITRINE_STATIC}")
        set(_vitrine_kind STATIC)
    elseif(EXISTS "${VITRINE_SHARED}")
        set(_vitrine_lib "${VITRINE_SHARED}")
        set(_vitrine_kind SHARED)
    else()
        message(FATAL_ERROR
            "vitrine: found the package config at ${CMAKE_CURRENT_LIST_DIR} but no library at "
            "${VITRINE_STATIC} or ${VITRINE_SHARED}. Run `cargo build --release` first, or install "
            "the crate's artefacts under this prefix.")
    endif()

    set_target_properties(vitrine::vitrine PROPERTIES
        IMPORTED_LOCATION "${_vitrine_lib}"
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        INTERFACE_INCLUDE_DIRECTORIES "${VITRINE_INCLUDE_DIR}")

    if(_vitrine_kind STREQUAL "STATIC")
        # A Rust staticlib needs the platform's C runtime and, on Linux, libdl/libpthread/libm.
        find_package(Threads REQUIRED)
        set_target_properties(vitrine::vitrine PROPERTIES
            INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS};m")
    endif()

    if(NOT EXISTS "${VITRINE_INCLUDE_DIR}/vitrine.h")
        message(FATAL_ERROR "vitrine: no vitrine.h under ${VITRINE_INCLUDE_DIR}")
    endif()

    set(vitrine_FOUND TRUE)
    set(VITRINE_FOUND TRUE)
endif()
