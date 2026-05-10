# ==========================================
# KNOWLEDGE GRAPH
# ==========================================

GRAPH = {

    "Road": {
        "safe": True,
        "flat": True,
        "risk": "Low",
        "preferred": True
    },

    "Grass": {
        "safe": True,
        "flat": False,
        "risk": "Medium",
        "preferred": False
    },

    "Vegetation": {
        "safe": False,
        "flat": False,
        "risk": "High",
        "preferred": False
    },

    "Water": {
        "safe": False,
        "flat": False,
        "risk": "Very High",
        "preferred": False
    },

    "Obstacle": {
        "safe": False,
        "flat": False,
        "risk": "Very High",
        "preferred": False
    },

    "Building": {
        "safe": False,
        "flat": True,
        "risk": "High",
        "preferred": False
    },

    "Tree": {
        "safe": False,
        "flat": False,
        "risk": "High",
        "preferred": False
    }
}


# ==========================================
# KNOWLEDGE GRAPH REASONING
# ==========================================

def think(terrain, roughness, distance):

    if terrain not in GRAPH:

        return {
            "decision": "Unknown",
            "risk": "Unknown",
            "score": 999,
            "mission_factors": "Unknown Terrain"
        }

    info = GRAPH[terrain]

    score = 0

    mission_factors = []


    # ==========================================
    # TERRAIN SAFETY
    # ==========================================

    if info["safe"]:

        score += 10

        mission_factors.append("Safe Terrain")

    else:

        score += 100

        mission_factors.append("Unsafe Terrain")


    # ==========================================
    # FLATNESS
    # ==========================================

    if roughness < 0.05:

        score += 5

        mission_factors.append("Flat Surface")

    else:

        score += 40

        mission_factors.append("Rough Terrain")


    # ==========================================
    # TARGET DISTANCE
    # ==========================================

    if distance < 150:

        score += 5

        mission_factors.append("Near Target")

    else:

        score += 20

        mission_factors.append("Far From Target")


    # ==========================================
    # PREFERRED TERRAIN
    # ==========================================

    if info["preferred"]:

        score -= 10

        mission_factors.append("Preferred Landing Surface")


    # ==========================================
    # FINAL DECISION
    # ==========================================

    if score < 20:

        decision = "Recommended Landing Zone"

    elif score < 50:

        decision = "Moderately Safe"

    else:

        decision = "Unsafe Landing Zone"


    return {

        "decision": decision,

        "risk": info["risk"],

        "score": score,

        "mission_factors": ", ".join(mission_factors)
    }