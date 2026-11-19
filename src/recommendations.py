"""
EcoSort AI - Waste Segregation & Recommendation Engine
Provides decoupled two-dustbin segregation recommendations, step-by-step
disposal procedures, material science profiles, decomposition timelines,
contamination warnings, and quantified environmental impact metrics.
"""

from typing import Dict, Any, List

# Standard two-dustbin types
BIN_BLUE = "Blue – Dry Waste"
BIN_GREEN = "Green – Wet / Biodegradable Waste"
BIN_VERIFY = "User Verification Required"

# Comprehensive Category Metadata, Material Science, and Segregation Database
DISPOSAL_DATABASE: Dict[str, Dict[str, Any]] = {
    "cardboard": {
        "category_title": "Cardboard",
        "dustbin": BIN_BLUE,
        "dustbin_color": "#1d4ed8",  # Rich Blue
        "bin_type": "dry",
        "bin_icon": "🟦",
        "badge_color": "#2563eb",
        "verification_required": False,
        "decomposition_time": "2 to 3 months (in optimal aerobic composting)",
        "material_profile": "Corrugated cellulose fiber board manufactured via the Kraft chemical pulping process. Features high tensile strength, multiple unbleached flute layers, and high biodegradable cellulose content.",
        "accepted_examples": [
            "Corrugated shipping cartons & delivery boxes",
            "Cereal, snack, and dry grocery product boxes",
            "Egg cartons (clean, unsoiled paper pulp)",
            "Cardboard tubes, cores, and packing inserts",
            "Clean paperboard gift and retail packaging"
        ],
        "rejected_examples": [
            "Greasy pizza boxes with heavy melted cheese or oil stains (contaminates repulping slurry)",
            "Wax-coated, plastic-laminated, or waterproofed frozen food boxes",
            "Boxes saturated with liquids, paints, solvents, or food grease",
            "Cardboard with glued styrofoam or synthetic bubble wrap still attached"
        ],
        "instructions": [
            "Remove unwanted packaging (e.g., plastic tape, styrofoam, bubble wrap).",
            "Ensure it is completely clean and dry.",
            "Flatten the box or corrugated sheet to save space.",
            "Place it neatly in the blue bin for dry recyclable waste."
        ],
        "detailed_steps": [
            "Inspect and strip all external non-cardboard contaminants, including synthetic adhesive tape, shipping label pouches, staples, and internal plastic/styrofoam cushions.",
            "Ensure the cardboard is dry and free of oil, cheese, butter, or liquid food stains (oil prevents fiber bonding during mechanical repulping).",
            "Break down corner seams and flatten boxes completely to maximize collection volume and reduce municipal transport emissions.",
            "Place neatly inside the BLUE DUSTBIN or bundle together with natural twine if volume exceeds bin capacity."
        ],
        "recycling_process": "At the Material Recovery Facility (MRF), cardboard is sorted, compacted into multi-ton bales, and transported to paper mills. It is placed in a hydro-pulper with warm water to break down into pulp slurry, screened to remove ink and glue, and rolled into new recycled packaging sheets.",
        "environmental_savings": {
            "energy_saved": "24% less energy compared to virgin fiber manufacturing",
            "trees_saved": "17 mature trees preserved per metric ton recycled",
            "water_saved": "7,000 gallons (26,500 L) of water conserved per ton",
            "landfill_saved": "9 cubic yards of landfill volume spared per ton",
            "ghg_reduction": "Approx. 1.0 metric ton of CO₂ equivalent avoided per ton"
        },
        "environmental_tip": "Recycling one ton of cardboard saves up to 9 cubic yards of landfill space and 24% of the energy required for virgin cardboard production.",
        "recyclability": "High (100% Recyclable when clean & unsoiled)"
    },
    "glass": {
        "category_title": "Glass",
        "dustbin": BIN_BLUE,
        "dustbin_color": "#1d4ed8",
        "bin_type": "dry",
        "bin_icon": "🟦",
        "badge_color": "#0284c7",
        "verification_required": False,
        "decomposition_time": "1,000,000+ years (practically indefinite under ambient conditions)",
        "material_profile": "Non-crystalline amorphous solid composed primarily of silicon dioxide (silica sand, SiO₂), sodium carbonate (soda ash, Na₂CO₃), and calcium carbonate (limestone, CaCO₃). Thermodynamically stable and chemically inert.",
        "accepted_examples": [
            "Clear (flint), green (emerald), and amber (brown) beverage bottles",
            "Food jars (jam, pasta sauce, condiment, mayonnaise jars)",
            "Cosmetic and pharmaceutical glass bottles (rinsed clean)",
            "Container glassware with standard soda-lime formulation"
        ],
        "rejected_examples": [
            "Pyrex, borosilicate, or heat-resistant cookware (melts at much higher temperatures, causing batch defects)",
            "Window panes, automotive windshields, and laminated architectural glass",
            "Mirrors (contain reflective chemical backing coats)",
            "Ceramics, porcelain, pottery, or earthenware",
            "Incandescent or fluorescent light bulbs (contain toxic phosphors or mercury)"
        ],
        "instructions": [
            "Empty any liquid or solid contents from the container.",
            "Check whether the bottle or jar can be reused or returned.",
            "Handle broken glass carefully, wrap it securely in thick paper or cardboard.",
            "Follow local municipal glass collection rules and place in the blue dry-waste bin."
        ],
        "detailed_steps": [
            "Completely drain liquids and scrape out all food residue from jars and bottles.",
            "Rinse briefly with minimal water to eliminate odors and bacterial growth.",
            "Remove metal or plastic caps and lids (place metal lids in blue bin separately; remove cork stoppers).",
            "If glass is broken, wrap shards securely in several layers of thick newspaper or cardboard to protect sanitation workers from severe lacerations, then deposit in the BLUE DUSTBIN."
        ],
        "recycling_process": "Collected container glass is color-sorted (clear, green, amber), crushed into granular cullet, and blended with virgin raw materials. Cullet melts at a significantly lower furnace temperature than raw silica sand (saving up to 30% furnace energy), where it is endlessly blown or pressed into brand new bottles and jars without any loss of structural integrity.",
        "environmental_savings": {
            "energy_saved": "30% reduction in glass smelting furnace energy consumption",
            "raw_minerals_saved": "1.2 tons of virgin silica, soda ash, and limestone spared per ton recycled",
            "air_pollution": "20% reduction in sulfur dioxide and nitrogen oxide emissions",
            "co2_reduction": "315 kg of CO₂ avoided for every ton of cullet used in the kiln",
            "cycle_efficiency": "Can return to grocery store shelves as a new container within 30 days"
        },
        "environmental_tip": "Glass is infinitely recyclable without losing its quality, strength, or clarity. A recycled glass container saves enough energy to power a light bulb for 4 hours.",
        "recyclability": "High (100% Endlessly Recyclable)"
    },
    "metal": {
        "category_title": "Metal",
        "dustbin": BIN_BLUE,
        "dustbin_color": "#1d4ed8",
        "bin_type": "dry",
        "bin_icon": "🟦",
        "badge_color": "#475569",
        "verification_required": False,
        "decomposition_time": "50 to 500 years (Aluminum: 200-500 years; Tin/Steel: 50-100 years)",
        "material_profile": "Comprises non-ferrous aluminum alloys (Al-Mg-Mn series for cans) and ferrous tin-plated steel (tinplate food cans). Characterized by metallic crystalline lattices, high thermal conductivity, and high resale economic value.",
        "accepted_examples": [
            "Aluminum carbonated beverage and soda cans",
            "Tin and steel food preservation cans (soup, tuna, vegetable cans)",
            "Clean aluminum baking foil and takeaway food trays",
            "Metal bottle crown caps, jar lids, and twist-off closures",
            "Empty aerosol spray cans (completely depressurized and emptied)"
        ],
        "rejected_examples": [
            "Pressurized aerosol cans containing flammable propellants or pesticides",
            "Paint, varnish, or industrial chemical cans with dried toxic residue",
            "Scrap motor vehicle parts, car batteries, or lead/zinc alloys",
            "Electronic scrap, copper wiring, or heavy industrial machinery parts",
            "Propane gas cylinders or compressed gas canisters (explosion hazard)"
        ],
        "instructions": [
            "Empty the container thoroughly of any liquid or food contents.",
            "Remove any food residue or labels if necessary.",
            "Rinse briefly with water if dirty.",
            "Place it in the blue dry-waste bin if accepted by your local recycler."
        ],
        "detailed_steps": [
            "Ensure the metal can or tray is fully emptied of food or beverage contents.",
            "Perform a swift rinse to remove fats, sauces, or sugar syrups.",
            "Push the metal lid inside the tin can or crimp the top slightly so the lid cannot fall out or present exposed sharp edges.",
            "Crush aluminum cans if convenient to conserve bin space, and deposit in the BLUE DUSTBIN."
        ],
        "recycling_process": "At sorting hubs, ferrous steel is separated using industrial electro-magnets, while non-ferrous aluminum is segregated using eddy-current magnetic separators. Metals are shredded, decoating kilns strip lacquers and paints, and the shreds are smelted into ingots for rolling into fresh automotive sheets, construction materials, and new beverage cans.",
        "environmental_savings": {
            "energy_saved": "95% energy saved for aluminum; 75% energy saved for steel vs virgin mining",
            "mining_spared": "4 tons of bauxite ore spared per single ton of recycled aluminum",
            "water_reduction": "40% reduction in water usage compared to mining raw iron and bauxite",
            "co2_avoided": "9 tons of CO₂ emissions prevented per metric ton of aluminum recycled",
            "economic_value": "Metal scrap is the highest economic value recyclable stream in municipal systems"
        },
        "environmental_tip": "Recycling aluminum cans saves 95% of the energy required to make new ones from raw bauxite ore. Aluminum can be recycled back into a new can within 60 days.",
        "recyclability": "High (Highly Recyclable & Extremely Valuable)"
    },
    "paper": {
        "category_title": "Paper",
        "dustbin": BIN_BLUE,
        "dustbin_color": "#1d4ed8",
        "bin_type": "dry",
        "bin_icon": "🟦",
        "badge_color": "#0ea5e9",
        "verification_required": False,
        "decomposition_time": "2 to 6 weeks (in aerobic conditions; decades if buried compacted in anaerobic landfills)",
        "material_profile": "Thin sheet material produced by mechanically or chemically processing virgin softwood/hardwood cellulose fibers. Sensitive to moisture, oil, and grease contamination which disrupt hydrogen bonding between cellulose fibrils.",
        "accepted_examples": [
            "White and colored office printer and copier paper",
            "Newspapers, newsprint supplements, and magazines (glossy and matte)",
            "Envelopes (including those with clear cellophane address windows)",
            "Notebooks, softcover books, school exercise books, and paper manuals",
            "Clean paper bags, wrapping sheets, and paper flyers"
        ],
        "rejected_examples": [
            "Used paper napkins, facial tissues, and kitchen paper towels (fibers are too short and often soiled)",
            "Grease-stained takeaway paper wraps and buttery parchment baking sheets",
            "Waxed paper, silicone release liners, and waterproof coated butcher paper",
            "Thermal paper cash register receipts (contain endocrine-disrupting Bisphenol-A / BPA coatings)",
            "Composite foil-backed or plastic-laminated wrapping paper"
        ],
        "instructions": [
            "Check that the paper is clean and dry (not soiled by oil or grease).",
            "Separate it from any food waste, wet wrappers, or plastic coating.",
            "Fold or flatten it neatly.",
            "Place it in the blue bin for dry recyclables."
        ],
        "detailed_steps": [
            "Perform a tactile moisture check: ensure paper is entirely dry and free from grease, oil, or food spills.",
            "Remove plastic shrink-wraps, metallic binder rings, and thick plastic sleeves (standard paper staples and envelope windows are acceptable at modern repulping plants).",
            "Flatten and fold documents neatly to prevent loose sheets from scattering during collection.",
            "Place into the BLUE DUSTBIN for clean dry recyclables."
        ],
        "recycling_process": "Collected clean paper is graded by fiber length (white ledger, mixed office, newsprint). At the paper mill, it is agitated in warm water into a pulp, passed through flotation de-inking cells to remove print pigments, bleached using chlorine-free hydrogen peroxide, and laid onto fast-moving wire screens to produce recycled newsprint, packaging, or tissue.",
        "environmental_savings": {
            "trees_saved": "17 mature trees saved per ton of recycled paper",
            "water_conserved": "7,000 gallons (26,500 L) of freshwater saved per ton",
            "electricity_saved": "4,100 kilowatt-hours (kWh) of electricity conserved per ton",
            "landfill_space": "3.3 cubic yards of municipal landfill space preserved",
            "air_emissions": "73% less air pollution compared to manufacturing virgin wood pulp"
        },
        "environmental_tip": "Every ton of paper recycled saves approximately 17 mature trees, 7,000 gallons of water, and 4,000 kilowatt-hours of electrical energy.",
        "recyclability": "High (Recyclable if clean, dry & unsoiled)"
    },
    "plastic": {
        "category_title": "Plastic",
        "dustbin": BIN_BLUE,
        "dustbin_color": "#1d4ed8",
        "bin_type": "dry",
        "bin_icon": "🟦",
        "badge_color": "#0369a1",
        "verification_required": False,
        "decomposition_time": "450 to 1,000 years (fragments into harmful micro- and nanoplastics without biological assimilation)",
        "material_profile": "Synthetic or semi-synthetic organic polymers derived from petrochemical hydrocarbons. Predominant types include Polyethylene Terephthalate (PET #1), High-Density Polyethylene (HDPE #2), Polypropylene (PP #5), and Low-Density Polyethylene (LDPE #4).",
        "accepted_examples": [
            "PET #1 beverage and water bottles, cooking oil bottles (rinsed)",
            "HDPE #2 milk jugs, shampoo, bodywash, and laundry detergent jugs",
            "PP #5 yogurt tubs, margarine tubs, and reusable food containers",
            "Clean rigid plastic packaging containers with identifiable SPI recycling resin codes",
            "Plastic bottle caps (screwed back tightly onto rinsed bottles)"
        ],
        "rejected_examples": [
            "Plastic shopping carrier bags and ultra-thin cling films (wrap around and jam MRF sorting gears)",
            "Expanded Polystyrene (EPS #6 / Styrofoam packaging, cups, insulation trays)",
            "PVC #3 pipes, medical blister packaging, and imitation leather",
            "Multi-layer foil-plastic metallized snack chip pouches and sachets",
            "Bio-plastics labeled PLA (compostable plastics contaminate conventional PET recycling streams)"
        ],
        "instructions": [
            "Empty all remaining liquids or food contents from the plastic item.",
            "Rinse quickly if sticky or greasy.",
            "Allow the plastic container to dry completely.",
            "Place it in the blue bin if accepted locally (check recycling resin codes #1-#7)."
        ],
        "detailed_steps": [
            "Verify the resin code stamp on the underside (PET #1 and HDPE #2 are accepted globally; PP #5 is widely accepted).",
            "Completely drain residual beverages, detergents, or cosmetics.",
            "Rinse with minimal water and shake dry to prevent bacterial growth and bin contamination.",
            "Screw the plastic cap back onto the bottle (caps are captured during mechanical flaking and density float-sink tanks), flatten the bottle body, and deposit in the BLUE DUSTBIN."
        ],
        "recycling_process": "At the plastics reclamation facility, NIR (Near-Infrared) optical sensors categorize bottles by polymer type. Items are shredded into flakes, washed in hot caustic baths to dissolve adhesives and labels, passed through float-sink density separation tanks, dried, and melted through extruders into pure recycled pellets (rPET, rHDPE) for textile yarns, bottles, or industrial piping.",
        "environmental_savings": {
            "oil_saved": "16.3 barrels (2,500 L) of crude petroleum oil conserved per ton of recycled plastic",
            "energy_saved": "5,774 kWh of electricity saved compared to synthesizing virgin polymers",
            "ocean_protection": "Prevents plastic ingestion and entanglement by marine life; stops microplastic proliferation",
            "landfill_spared": "30 cubic yards of municipal landfill space conserved per ton",
            "carbon_savings": "Approx. 1.5 to 2.0 metric tons of greenhouse gas emissions eliminated per ton"
        },
        "environmental_tip": "Segregating plastic items prevents microplastic ocean contamination and reduces demand for virgin petrochemical feedstocks.",
        "recyclability": "Moderate (Depends on resin polymer code; PET #1 & HDPE #2 are widely recycled)"
    },
    "trash": {
        "category_title": "Trash / Residual Waste",
        "dustbin": BIN_VERIFY,
        "dustbin_color": "#b45309",  # Amber/Ochre for verification
        "bin_type": "verification",
        "bin_icon": "⚠️",
        "badge_color": "#d97706",
        "verification_required": True,
        "decomposition_time": "Varies: 2 weeks (organic food peels) to centuries (synthetic composites & sanitary products)",
        "material_profile": "Heterogeneous category encompassing complex mixed-material composites, soiled municipal discards, organic food remains, sanitary waste, or non-recyclable residual materials that cannot be mechanically sorted into single streams.",
        "accepted_examples": [
            "Biodegradable food scraps, fruit peels, vegetable ends, egg shells, coffee grounds",
            "Soiled takeaway wrappers, used hygiene tissues, grease-stained packaging",
            "Multi-laminated metallic snack sachets (aluminum bonded with polymer films)",
            "Worn-out textiles, broken synthetic ceramics, non-recyclable composite items",
            "Broken toys, cigarette butts, vacuum dust, sweepings"
        ],
        "rejected_examples": [
            "Clean dry cardboard, paper, glass, metal, or rigid plastic (MUST go to the Blue Bin)",
            "Rechargeable lithium-ion batteries and mobile phone electronics (Fire Hazard)",
            "Hazardous solvents, biomedical needles, prescription pharmaceuticals, or motor oils",
            "Corrosive chemicals, liquid paints, industrial pesticides"
        ],
        "instructions": [
            "Inspect the item.",
            "Determine whether it is biodegradable or dry residual waste.",
            "Select the appropriate disposal route.",
            "Follow local rules for hazardous or special waste."
        ],
        "detailed_steps": [
            "CRITICAL: The deep learning model does NOT possess a standalone wet/organic class. Never automatically toss every Trash item into the green bin.",
            "Inspect the physical item to determine its fundamental nature:",
            "  • If BIODEGRADABLE ORGANIC (food scraps, fruit/vegetable peels, wilted flowers, tea bags): Place into the GREEN DUSTBIN for composting or biogas generation.",
            "  • If DRY NON-RECYCLABLE RESIDUE (multi-layer laminate sachets, soiled wrappers, thermal paper, broken composite items): Place into designated MUNICIPAL RESIDUAL DRY WASTE collection.",
            "  • If HAZARDOUS, ELECTRONIC, or SANITARY WASTE: Strictly isolate and hand over to certified municipal e-waste or hazardous collection drives. Do NOT combine with household waste."
        ],
        "recycling_process": "Municipal residual waste that is properly segregated allows organic wet waste to undergo aerobic composting or anaerobic digestion (producing organic fertilizer and methane biogas). Non-recyclable dry residues are sent to modern Waste-to-Energy (WtE) incineration plants for electricity generation or engineered sanitary landfills with leachate and biogas capture.",
        "environmental_savings": {
            "methane_abatement": "Diverting wet organic waste from landfills eliminates fugitive methane emissions (CH₄ is 28x more potent than CO₂)",
            "soil_enrichment": "Organic compost restores nitrogen, phosphorus, and organic carbon to agricultural soils",
            "purity_preservation": "Keeping wet items out of the Blue Bin prevents degradation of hundreds of tons of recyclable paper and plastic",
            "landfill_longevity": "Extends municipal landfill lifespans by up to 40% through strict source diversion"
        },
        "environmental_tip": "Segregation at source ensures that biodegradable waste can be composted without chemical contamination, while non-biodegradable waste is processed safely.",
        "recyclability": "Variable (Requires manual inspection and verification)"
    }
}


def get_recommendation(category: str) -> Dict[str, Any]:
    """
    Returns the comprehensive segregation recommendation, dustbin color,
    detailed step-by-step instructions, material profile, and environmental metrics.
    """
    clean_cat = category.strip().lower()
    if clean_cat not in DISPOSAL_DATABASE:
        raise ValueError(f"Unknown waste category: '{category}'. Valid: {list(DISPOSAL_DATABASE.keys())}")
    return DISPOSAL_DATABASE[clean_cat]


def list_supported_categories() -> List[str]:
    """Returns the list of 6 supported categories."""
    return list(DISPOSAL_DATABASE.keys())
