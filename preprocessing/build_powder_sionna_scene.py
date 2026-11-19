import json
import math
import csv
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path.home() / "ndt_modeling_benchmark"

GEOJSON = ROOT / "data/osm/powder_buildings.geojson"

SCENE_DIR = ROOT / "models/hybrid_sionna_lena/scene"
MESH_DIR = SCENE_DIR / "meshes"

SCENE_DIR.mkdir(parents=True, exist_ok=True)
MESH_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# Benchmark coordinate system
# ----------------------------------------------------------

LAT0 = 40.76855
LON0 = -111.84616

# IMPORTANT:
# This is the nominal benchmark gNB origin.
# gNB radio position will later be (0,0,10).
R_EARTH = 6371000.0

# ----------------------------------------------------------
# Scene bounds
#
# Trajectory roughly:
# x: -107 .. 306 m
# y:   52 .. 296 m
#
# Add generous margins around trajectory and MEB.
# ----------------------------------------------------------

X_MIN = -180.0
X_MAX = 380.0

Y_MIN = -120.0
Y_MAX = 380.0

GROUND_Z = 0.0

DEFAULT_HEIGHT = 9.0
DEFAULT_LEVEL_HEIGHT = 3.0

# ----------------------------------------------------------
# Coordinate transform
# ----------------------------------------------------------

def latlon_to_local(lat, lon):
    lat0_rad = math.radians(LAT0)

    x = (
        R_EARTH
        * math.radians(lon - LON0)
        * math.cos(lat0_rad)
    )

    y = (
        R_EARTH
        * math.radians(lat - LAT0)
    )

    return x, y


# ----------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------

def signed_area(poly):
    a = 0.0

    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]

        a += x1 * y2 - x2 * y1

    return 0.5 * a


def clean_ring(ring):
    pts = []

    for lon, lat in ring:
        x, y = latlon_to_local(lat, lon)

        if not pts or abs(x - pts[-1][0]) > 1e-8 or abs(y - pts[-1][1]) > 1e-8:
            pts.append((x, y))

    # GeoJSON polygons usually repeat first vertex
    if len(pts) >= 2:
        if abs(pts[0][0] - pts[-1][0]) < 1e-8 and abs(pts[0][1] - pts[-1][1]) < 1e-8:
            pts.pop()

    return pts


def polygon_intersects_scene(poly):
    if not poly:
        return False

    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]

    return not (
        max(xs) < X_MIN
        or min(xs) > X_MAX
        or max(ys) < Y_MIN
        or min(ys) > Y_MAX
    )


# ----------------------------------------------------------
# Ear clipping triangulation
#
# Handles normal simple polygons.
# OSM building footprints are usually simple polygons.
# ----------------------------------------------------------

def point_in_triangle(p, a, b, c):
    def sign(p1, p2, p3):
        return (
            (p1[0] - p3[0]) * (p2[1] - p3[1])
            - (p2[0] - p3[0]) * (p1[1] - p3[1])
        )

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def is_convex(a, b, c, ccw):
    cross = (
        (b[0] - a[0]) * (c[1] - b[1])
        - (b[1] - a[1]) * (c[0] - b[0])
    )

    return cross > 0 if ccw else cross < 0


def triangulate_polygon(poly):
    n = len(poly)

    if n < 3:
        return []

    ccw = signed_area(poly) > 0

    indices = list(range(n))
    triangles = []

    guard = 0

    while len(indices) > 3 and guard < 10000:
        guard += 1
        ear_found = False

        for k in range(len(indices)):
            i_prev = indices[(k - 1) % len(indices)]
            i_curr = indices[k]
            i_next = indices[(k + 1) % len(indices)]

            a = poly[i_prev]
            b = poly[i_curr]
            c = poly[i_next]

            if not is_convex(a, b, c, ccw):
                continue

            contains = False

            for j in indices:
                if j in (i_prev, i_curr, i_next):
                    continue

                if point_in_triangle(poly[j], a, b, c):
                    contains = True
                    break

            if contains:
                continue

            if ccw:
                triangles.append((i_prev, i_curr, i_next))
            else:
                triangles.append((i_next, i_curr, i_prev))

            del indices[k]

            ear_found = True
            break

        if not ear_found:
            # Fallback fan triangulation
            triangles = []

            for i in range(1, n - 1):
                triangles.append((0, i, i + 1))

            return triangles

    if len(indices) == 3:
        if ccw:
            triangles.append(tuple(indices))
        else:
            triangles.append(
                (indices[2], indices[1], indices[0])
            )

    return triangles


# ----------------------------------------------------------
# Height extraction
# ----------------------------------------------------------

def parse_float(v):
    if v is None:
        return None

    try:
        return float(v)
    except Exception:
        pass

    # e.g. "12 m"
    try:
        return float(str(v).lower().replace("meters", "").replace("meter", "").replace("m", "").strip())
    except Exception:
        return None


def building_height(props):
    # Your GeoJSON already contains cleaned height_est_m
    for key in [
        "height_est_m",
        "height_osm_m",
        "height",
        "height_from_levels_m",
    ]:
        h = parse_float(props.get(key))

        if h is not None and h > 0:
            return h, key

    levels = parse_float(props.get("building:levels"))

    if levels is None:
        levels = parse_float(props.get("levels_numeric"))

    if levels is not None and levels > 0:
        return levels * DEFAULT_LEVEL_HEIGHT, "levels_fallback"

    return DEFAULT_HEIGHT, "default_9m"


# ----------------------------------------------------------
# OBJ writer
# ----------------------------------------------------------

vertices = []
faces = []


def add_vertex(x, y, z):
    vertices.append((x, y, z))
    return len(vertices)  # OBJ is 1-indexed


def add_building(poly, height):
    n = len(poly)

    if n < 3:
        return

    bottom = []
    top = []

    for x, y in poly:
        bottom.append(add_vertex(x, y, GROUND_Z))

    for x, y in poly:
        top.append(add_vertex(x, y, height))

    ccw = signed_area(poly) > 0

    # Walls
    for i in range(n):
        j = (i + 1) % n

        b1 = bottom[i]
        b2 = bottom[j]
        t1 = top[i]
        t2 = top[j]

        if ccw:
            faces.append((b1, b2, t2))
            faces.append((b1, t2, t1))
        else:
            faces.append((b1, t2, b2))
            faces.append((b1, t1, t2))

    # Roof
    triangles = triangulate_polygon(poly)

    for a, b, c in triangles:
        faces.append(
            (
                top[a],
                top[b],
                top[c],
            )
        )


# ----------------------------------------------------------
# Load GeoJSON
# ----------------------------------------------------------

with GEOJSON.open() as f:
    data = json.load(f)

report = []

used = 0
skipped = 0

for idx, feature in enumerate(data["features"]):

    props = feature.get("properties", {})
    geom = feature.get("geometry", {})

    geom_type = geom.get("type")
    coords = geom.get("coordinates")

    polygons = []

    if geom_type == "Polygon":
        polygons = [coords]

    elif geom_type == "MultiPolygon":
        polygons = coords

    else:
        continue

    height, height_source = building_height(props)

    building_name = (
        props.get("name")
        or props.get("short_name")
        or f"building_{idx}"
    )

    for polygon_idx, polygon in enumerate(polygons):

        if not polygon:
            continue

        # Exterior ring only.
        # Holes are ignored because they are rare for these OSM footprints.
        exterior = polygon[0]

        poly = clean_ring(exterior)

        if len(poly) < 3:
            skipped += 1
            continue

        if not polygon_intersects_scene(poly):
            continue

        before_v = len(vertices)
        before_f = len(faces)

        add_building(poly, height)

        used += 1

        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]

        report.append(
            {
                "feature_index": idx,
                "polygon_index": polygon_idx,
                "name": building_name,
                "osm_id": props.get("id"),
                "height_m": height,
                "height_source": height_source,
                "x_min": min(xs),
                "x_max": max(xs),
                "y_min": min(ys),
                "y_max": max(ys),
                "vertices_added": len(vertices) - before_v,
                "faces_added": len(faces) - before_f,
            }
        )


# ----------------------------------------------------------
# Write buildings OBJ
# ----------------------------------------------------------

BUILDINGS_OBJ = MESH_DIR / "powder_buildings.obj"

with BUILDINGS_OBJ.open("w") as f:

    f.write("# POWDER / MEB OSM building scene\n")
    f.write("# Local ENU-like coordinates relative to nominal MEB gNB\n")

    for x, y, z in vertices:
        f.write(
            f"v {x:.6f} {y:.6f} {z:.6f}\n"
        )

    for a, b, c in faces:
        f.write(
            f"f {a} {b} {c}\n"
        )


# ----------------------------------------------------------
# Ground OBJ
# ----------------------------------------------------------

GROUND_OBJ = MESH_DIR / "powder_ground.obj"

ground_vertices = [
    (X_MIN, Y_MIN, 0.0),
    (X_MAX, Y_MIN, 0.0),
    (X_MAX, Y_MAX, 0.0),
    (X_MIN, Y_MAX, 0.0),
]

with GROUND_OBJ.open("w") as f:

    f.write("# POWDER ground plane\n")

    for x, y, z in ground_vertices:
        f.write(
            f"v {x:.6f} {y:.6f} {z:.6f}\n"
        )

    f.write("f 1 2 3\n")
    f.write("f 1 3 4\n")


# ----------------------------------------------------------
# Mitsuba / Sionna XML
# ----------------------------------------------------------

XML = SCENE_DIR / "powder_meb.xml"

xml = f'''<scene version="2.1.0">

    <!-- Radio materials -->

    <bsdf type="itu-radio-material" id="building-concrete">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.20"/>
    </bsdf>

    <bsdf type="itu-radio-material" id="ground-asphalt">
        <string name="type" value="asphalt_concrete"/>
        <float name="thickness" value="0.20"/>
    </bsdf>

    <!-- OSM buildings -->

    <shape type="obj" id="powder-buildings">
        <string name="filename" value="meshes/powder_buildings.obj"/>
        <boolean name="face_normals" value="true"/>
        <ref id="building-concrete" name="bsdf"/>
    </shape>

    <!-- Ground / parking / roads approximation -->

    <shape type="obj" id="powder-ground">
        <string name="filename" value="meshes/powder_ground.obj"/>
        <boolean name="face_normals" value="true"/>
        <ref id="ground-asphalt" name="bsdf"/>
    </shape>

</scene>
'''

XML.write_text(xml)


# ----------------------------------------------------------
# Report
# ----------------------------------------------------------

REPORT = SCENE_DIR / "scene_buildings_report.csv"

if report:

    with REPORT.open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=report[0].keys()
        )

        writer.writeheader()
        writer.writerows(report)


# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

print("\n====================================")
print("POWDER SIONNA SCENE GENERATED")
print("====================================")

print("GeoJSON buildings:", len(data["features"]))
print("Polygons used:", used)
print("Skipped invalid polygons:", skipped)

print("Mesh vertices:", len(vertices))
print("Mesh triangles:", len(faces))

print("\nScene bounds:")
print("x:", X_MIN, "->", X_MAX, "m")
print("y:", Y_MIN, "->", Y_MAX, "m")

print("\nNominal benchmark gNB:")
print("(x,y,z) = (0,0,10 m)")

print("\nUE height:")
print("z = 1.5 m")

print("\nFiles:")
print(" ", BUILDINGS_OBJ)
print(" ", GROUND_OBJ)
print(" ", XML)
print(" ", REPORT)
