"""Word lists for the iGSM-style generator.

Each "world" is a 4-layer hierarchy of categories (e.g. School -> Classroom ->
Backpack -> Stationery). A problem picks one world and a few items from every
layer. Instance parameters are edges between consecutive layers ("the number of
each Central High 's Film Studio"), abstract parameters count a whole category
below an item ("Central High 's Backpack").
"""

WORLDS = [
    {
        "name": "school",
        "layers": [
            ("School", ["Central High", "Riverside Academy", "Oak Ridge School", "Maple Elementary",
                        "Pine Valley High", "Lakeside Middle", "Hilltop Prep", "Sunset Academy",
                        "Westwood High", "Brookfield School", "Cedar Grove High", "Northgate Academy"]),
            ("Classroom", ["Film Studio", "Dance Studio", "Science Lab", "Art Room", "Music Room",
                           "Computer Lab", "Chemistry Lab", "Reading Room", "Drama Studio",
                           "Math Room", "History Room", "Robotics Lab"]),
            ("Backpack", ["School Daypack", "Messenger Backpack", "Hiking Backpack", "Rolling Backpack",
                          "Laptop Backpack", "Canvas Backpack", "Sling Backpack", "Mini Backpack",
                          "Drawstring Backpack", "Leather Backpack"]),
            ("Stationery", ["Pencil", "Eraser", "Ruler", "Marker", "Notebook", "Highlighter",
                            "Glue Stick", "Sharpener", "Crayon", "Stapler", "Compass", "Protractor"]),
        ],
    },
    {
        "name": "zoo",
        "layers": [
            ("Zoo", ["Riverview Zoo", "Sunny Hill Zoo", "Bayside Zoo", "Greenfield Zoo",
                     "Mountain Peak Zoo", "Desert Sands Zoo", "Lakeshore Zoo", "Forest Glen Zoo",
                     "Coral Bay Zoo", "Prairie Zoo", "Willow Creek Zoo", "Granite Zoo"]),
            ("Enclosure", ["Reptile House", "Bird Aviary", "Savanna Exhibit", "Primate Island",
                           "Polar Pavilion", "Nocturnal House", "Rainforest Dome", "Aquarium Hall",
                           "Insect House", "Petting Barn", "Wetland Trail", "Canyon Exhibit"]),
            ("Animal", ["Lion", "Zebra", "Parrot", "Python", "Penguin", "Giraffe", "Gorilla",
                        "Flamingo", "Tortoise", "Lemur", "Otter", "Cheetah", "Iguana", "Toucan"]),
            ("Bone", ["Femur", "Rib", "Skull", "Spine", "Pelvis", "Tibia", "Jawbone",
                      "Collarbone", "Kneecap", "Vertebra", "Ulna", "Humerus"]),
        ],
    },
    {
        "name": "city",
        "layers": [
            ("City", ["Springfield", "Rivertown", "Lakewood", "Fairview", "Kingston", "Ashford",
                      "Brighton", "Clearwater", "Dover", "Elmwood", "Glenville", "Harborview"]),
            ("Supermarket", ["Fresh Mart", "Value Grocer", "Corner Market", "Green Basket",
                             "Daily Foods", "Super Saver", "Harvest Market", "Family Grocer",
                             "Urban Pantry", "Golden Cart", "Prime Foods", "Market Square"]),
            ("Aisle", ["Dairy Aisle", "Bakery Aisle", "Snack Aisle", "Frozen Aisle", "Produce Aisle",
                       "Cereal Aisle", "Beverage Aisle", "Canned Aisle", "Spice Aisle", "Pasta Aisle"]),
            ("Product", ["Ice Cream", "Bagel", "Cheddar", "Pretzel", "Yogurt", "Granola", "Lemonade",
                         "Popcorn", "Oatmeal", "Cracker", "Muffin", "Butter", "Pickle", "Waffle"]),
        ],
    },
    {
        "name": "farm",
        "layers": [
            ("Farm", ["Green Acres", "Sunrise Farm", "Red Barn Ranch", "Golden Field Farm",
                      "Misty Valley Farm", "Oakwood Farm", "Silver Creek Ranch", "Meadowbrook Farm",
                      "Hollow Tree Farm", "Blue Sky Ranch", "Stony Brook Farm", "Windmill Farm"]),
            ("Barn", ["Hay Barn", "Dairy Barn", "Horse Stable", "Poultry Shed", "Tool Shed",
                      "Grain Silo", "Sheep Pen", "Goat Shelter", "Pig Sty", "Rabbit Hutch"]),
            ("Livestock", ["Cow", "Horse", "Chicken", "Sheep", "Goat", "Pig", "Duck", "Rabbit",
                           "Turkey", "Llama", "Donkey", "Alpaca"]),
            ("Tag", ["Ear Tag", "Leg Band", "Collar Tag", "Neck Chain", "Ankle Ring", "Tail Ribbon",
                     "Wing Clip", "Brand Mark", "Microchip", "Bell Collar"]),
        ],
    },
]

# Template words used in problem statements and solutions.
TEMPLATE_WORDS = [
    "The", "number", "of", "each", "equals", "more", "than", "times", "the", "sum",
    "difference", "product", "and", "Question", "How", "many", "does", "have",
    "Define", "as", "so", "Answer",
]

PUNCT = [".", ";", "=", "+", "-", "*", ":", "?", "'s"]

SPECIALS = ["<pad>", "<bos>", "<eos>", "<sol>", "<mask>", "<null>"]

MOD = 23

LETTERS = [chr(c) for c in range(ord("a"), ord("z") + 1)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def all_words():
    words = set()
    for w in WORLDS:
        for cat, items in w["layers"]:
            words.update(cat.split())
            for it in items:
                words.update(it.split())
    return words
