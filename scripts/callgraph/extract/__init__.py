"""extract/* modules each own one slice of node/edge production from a
ModuleScope + SourceFile. Nothing here mutates the GraphStore's adjacency
directly — they only call store.add_node / store.add_edge."""
