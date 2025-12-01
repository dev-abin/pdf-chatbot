import xml.etree.ElementTree as ET
import uuid
import re
from collections import defaultdict


# ----------------
# Helper functions
# ----------------

def canonical_path(node):
    """Return canonical XPath without indexes."""
    parts = []
    while node is not None:
        parts.append(node.tag.split("}")[-1])  # remove namespace if present
        node = node.getparent() if hasattr(node, "getparent") else None
    return "/" + "/".join(reversed(parts))


def infer_value_type(text):
    if text is None or not text.strip():
        return "empty"

    t = text.strip()

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        return "date"

    if re.fullmatch(r"[+-]?\d+", t):
        return "integer"

    if re.fullmatch(r"[+-]?\d+\.\d+", t):
        return "number"

    if re.fullmatch(r"(?i)true|false", t):
        return "boolean"

    if re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t):
        return "email"

    return "string"


def is_reference_node(node):
    """
    A node is considered a reference-only node if:
    - it has no child elements
    - AND it has an attribute named like refId, idRef, keyRef, etc.
    """

    has_children = any(isinstance(c.tag, str) for c in list(node))
    if has_children:
        return False

    ref_attrs = [a for a in node.attrib if "ref" in a.lower() or a.lower().endswith("id")]
    return len(ref_attrs) > 0


def collect_expected_children(struct_map):
    """
    If multiple documents (future extension), compute expected children per path.
    For now, expected = all children ever seen.
    """
    expected = {}
    for path, info in struct_map.items():
        expected[path] = sorted(info["children"])
    return expected


# ----------------
# Main fingerprint generator
# ----------------

def generate_xml_fingerprint(xml_string):
    root = ET.fromstring(xml_string)

    # track structure
    paths = {}
    occurrences = defaultdict(int)
    children_map = defaultdict(set)
    attr_map = defaultdict(set)

    reference_nodes = []     # for ref_model
    partial_nodes = []       # for missing/partial

    # iterative traversal
    stack = [(root, "/" + root.tag, 1)]
    node_paths_by_parent = defaultdict(list)

    while stack:
        node, path, depth = stack.pop()

        # count
        occurrences[path] += 1

        # determine node kind
        if is_reference_node(node):
            node_kind = "reference"
        else:
            node_kind = "inline"

        # collect children names
        child_elems = [c for c in list(node) if isinstance(c.tag, str)]
        for c in child_elems:
            children_map[path].add(c.tag)

        # collect attribute names + types only
        attr_types = {}
        for k, v in node.attrib.items():
            attr_types[k] = infer_value_type(v)

        attr_map[path] |= set(attr_types.keys())

        # detect repeated or single
        occurs = "repeated" if occurrences[path] > 1 else "single"

        # leaf value type
        text_val = (node.text or "").strip()
        leaf_type = infer_value_type(text_val) if (text_val and not child_elems) else "empty"

        # record path info once (first time)
        if path not in paths:
            paths[path] = {
                "path": path,
                "depth": depth,
                "node_kind": node_kind,
                "children": [],
                "attributes": {},
                "occurs": occurs,
                "value_type": leaf_type,
                "sample": text_val if text_val else None
            }
        else:
            # update dynamic fields
            paths[path]["node_kind"] = (
                "mixed" 
                if paths[path]["node_kind"] != node_kind 
                else node_kind
            )
            if leaf_type != "empty" and paths[path]["value_type"] == "empty":
                paths[path]["value_type"] = leaf_type
                paths[path]["sample"] = text_val

        # record attributes on this path
        for a in attr_types:
            paths[path]["attributes"][a] = attr_types[a]

        # reference node registration
        if node_kind == "reference":
            for a in node.attrib:
                if "ref" in a.lower():
                    reference_nodes.append({
                        "path": path,
                        "target_type": path.split("/")[-1].replace("Ref", ""),
                        "ref_attribute": a,
                        "mode": "reference-only"
                    })

        # push children into stack
        for c in child_elems:
            child_path = path + "/" + c.tag
            stack.append((c, child_path, depth + 1))

    # assign children lists
    for path, info in paths.items():
        info["children"] = sorted(children_map[path])
        info["attributes"] = {
            k: infer_value_type(k) if isinstance(info["attributes"], dict) else info["attributes"]
            for k in info["attributes"]
        }

    # detect partial definitions
    expected_children = collect_expected_children(paths)
    for path, info in paths.items():
        seen = set(info["children"])
        exp = set(expected_children[path])
        if exp and seen != exp:
            partial_nodes.append({
                "path": path,
                "expected_children": sorted(exp),
                "seen_children": sorted(seen),
                "status": "partial"
            })

    # final schema
    fingerprint = {
        "fingerprint_id": str(uuid.uuid4()),
        "root": {
            "name": root.tag,
            "depth": 1
        },
        "structure": {
            "paths": list(paths.values()),
            "missing_or_partial": partial_nodes
        },
        "ref_model": {
            "reference_nodes": reference_nodes
        }
    }

    # small cleanup: ensure attributes have types not dictionary-of-dicts
    for p in fingerprint["structure"]["paths"]:
        attr_map_clean = {}
        for aname, _ in p["attributes"].items():
            # Only store NAME → TYPE (correct behaviour)
            attr_map_clean[aname] = infer_value_type(node.attrib.get(aname, ""))
        p["attributes"] = attr_map_clean

    return fingerprint
