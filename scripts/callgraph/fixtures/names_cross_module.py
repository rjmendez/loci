"""Fixture: a cross-module READS_NAME — `names_cross_module_target.CONFIG_VALUE`
read through a module-level import, mirroring "ladybug_ops reading
server.MEMORY_DIR" (a read via an imported module, not a coincidence of
matching names)."""
import names_cross_module_target


def read_it():
    return names_cross_module_target.CONFIG_VALUE
