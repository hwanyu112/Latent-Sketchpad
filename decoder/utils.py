from PIL import Image, ImageDraw
import numpy as np
import random
from enum import Enum
class Category(Enum):
    """Enumerates the categories in the dataset 'Quick, Draw!'"""

    AIRCRAFT_CARRIER = "aircraft carrier"
    AIRPLANE = "airplane"
    ALARM_CLOCK = "alarm clock"
    AMBULANCE = "ambulance"
    ANGEL = "angel"
    ANIMAL_MIGRATION = "animal migration"
    ANT = "ant"
    ANVIL = "anvil"
    APPLE = "apple"
    ARM = "arm"
    ASPARAGUS = "asparagus"
    AXE = "axe"
    BACKPACK = "backpack"
    BANANA = "banana"
    BANDAGE = "bandage"
    BARN = "barn"
    BASEBALL = "baseball"
    BASEBALL_BAT = "baseball bat"
    BASKET = "basket"
    BASKETBALL = "basketball"
    BAT = "bat"
    BATHTUB = "bathtub"
    BEACH = "beach"
    BEAR = "bear"
    BEARD = "beard"
    BED = "bed"
    BEE = "bee"
    BELT = "belt"
    BENCH = "bench"
    BICYCLE = "bicycle"
    BINOCULARS = "binoculars"
    BIRD = "bird"
    BIRTHDAY_CAKE = "birthday cake"
    BLACKBERRY = "blackberry"
    BLUEBERRY = "blueberry"
    BOOK = "book"
    BOOMERANG = "boomerang"
    BOTTLECAP = "bottlecap"
    BOWTIE = "bowtie"
    BRACELET = "bracelet"
    BRAIN = "brain"
    BREAD = "bread"
    BRIDGE = "bridge"
    BROCCOLI = "broccoli"
    BROOM = "broom"
    BUCKET = "bucket"
    BULLDOZER = "bulldozer"
    BUS = "bus"
    BUSH = "bush"
    BUTTERFLY = "butterfly"
    CACTUS = "cactus"
    CAKE = "cake"
    CALCULATOR = "calculator"
    CALENDAR = "calendar"
    CAMEL = "camel"
    CAMERA = "camera"
    CAMOUFLAGE = "camouflage"
    CAMPFIRE = "campfire"
    CANDLE = "candle"
    CANNON = "cannon"
    CANOE = "canoe"
    CAR = "car"
    CARROT = "carrot"
    CASTLE = "castle"
    CAT = "cat"
    CEILING_FAN = "ceiling fan"
    CELLO = "cello"
    CELL_PHONE = "cell phone"
    CHAIR = "chair"
    CHANDELIER = "chandelier"
    CHURCH = "church"
    CIRCLE = "circle"
    CLARINET = "clarinet"
    CLOCK = "clock"
    CLOUD = "cloud"
    COFFEE_CUP = "coffee cup"
    COMPASS = "compass"
    COMPUTER = "computer"
    COOKIE = "cookie"
    COOLER = "cooler"
    COUCH = "couch"
    COW = "cow"
    CRAB = "crab"
    CRAYON = "crayon"
    CROCODILE = "crocodile"
    CROWN = "crown"
    CRUISE_SHIP = "cruise ship"
    CUP = "cup"
    DIAMOND = "diamond"
    DISHWASHER = "dishwasher"
    DIVING_BOARD = "diving board"
    DOG = "dog"
    DOLPHIN = "dolphin"
    DONUT = "donut"
    DOOR = "door"
    DRAGON = "dragon"
    DRESSER = "dresser"
    DRILL = "drill"
    DRUMS = "drums"
    DUCK = "duck"
    DUMBBELL = "dumbbell"
    EAR = "ear"
    ELBOW = "elbow"
    ELEPHANT = "elephant"
    ENVELOPE = "envelope"
    ERASER = "eraser"
    EYE = "eye"
    EYEGLASSES = "eyeglasses"
    FACE = "face"
    FAN = "fan"
    FEATHER = "feather"
    FENCE = "fence"
    FINGER = "finger"
    FIRE_HYDRANT = "fire hydrant"
    FIREPLACE = "fireplace"
    FIRETRUCK = "firetruck"
    FISH = "fish"
    FLAMINGO = "flamingo"
    FLASHLIGHT = "flashlight"
    FLIP_FLOPS = "flip flops"
    FLOOR_LAMP = "floor lamp"
    FLOWER = "flower"
    FLYING_SAUCER = "flying saucer"
    FOOT = "foot"
    FORK = "fork"
    FROG = "frog"
    FRYING_PAN = "frying pan"
    GARDEN = "garden"
    GARDEN_HOSE = "garden hose"
    GIRAFFE = "giraffe"
    GOATEE = "goatee"
    GOLF_CLUB = "golf club"
    GRAPES = "grapes"
    GRASS = "grass"
    GUITAR = "guitar"
    HAMBURGER = "hamburger"
    HAMMER = "hammer"
    HAND = "hand"
    HARP = "harp"
    HAT = "hat"
    HEADPHONES = "headphones"
    HEDGEHOG = "hedgehog"
    HELICOPTER = "helicopter"
    HELMET = "helmet"
    HEXAGON = "hexagon"
    HOCKEY_PUCK = "hockey puck"
    HOCKEY_STICK = "hockey stick"
    HORSE = "horse"
    HOSPITAL = "hospital"
    HOT_AIR_BALLOON = "hot air balloon"
    HOT_DOG = "hot dog"
    HOT_TUB = "hot tub"
    HOURGLASS = "hourglass"
    HOUSE = "house"
    HOUSE_PLANT = "house plant"
    HURRICANE = "hurricane"
    ICE_CREAM = "ice cream"
    JACKET = "jacket"
    JAIL = "jail"
    KANGAROO = "kangaroo"
    KEY = "key"
    KEYBOARD = "keyboard"
    KNEE = "knee"
    KNIFE = "knife"
    LADDER = "ladder"
    LANTERN = "lantern"
    LAPTOP = "laptop"
    LEAF = "leaf"
    LEG = "leg"
    LIGHT_BULB = "light bulb"
    LIGHTER = "lighter"
    LIGHTHOUSE = "lighthouse"
    LIGHTNING = "lightning"
    LINE = "line"
    LION = "lion"
    LIPSTICK = "lipstick"
    LOBSTER = "lobster"
    LOLLIPOP = "lollipop"
    MAILBOX = "mailbox"
    MAP = "map"
    MARKER = "marker"
    MATCHES = "matches"
    MEGAPHONE = "megaphone"
    MERMAID = "mermaid"
    MICROPHONE = "microphone"
    MICROWAVE = "microwave"
    MONKEY = "monkey"
    MOON = "moon"
    MOSQUITO = "mosquito"
    MOTORBIKE = "motorbike"
    MOUNTAIN = "mountain"
    MOUSE = "mouse"
    MOUSTACHE = "moustache"
    MOUTH = "mouth"
    MUG = "mug"
    MUSHROOM = "mushroom"
    NAIL = "nail"
    NECKLACE = "necklace"
    NOSE = "nose"
    OCEAN = "ocean"
    OCTAGON = "octagon"
    OCTOPUS = "octopus"
    ONION = "onion"
    OVEN = "oven"
    OWL = "owl"
    PAINTBRUSH = "paintbrush"
    PAINT_CAN = "paint can"
    PALM_TREE = "palm tree"
    PANDA = "panda"
    PANTS = "pants"
    PAPER_CLIP = "paper clip"
    PARACHUTE = "parachute"
    PARROT = "parrot"
    PASSPORT = "passport"
    PEANUT = "peanut"
    PEAR = "pear"
    PEAS = "peas"
    PENCIL = "pencil"
    PENGUIN = "penguin"
    PIANO = "piano"
    PICKUP_TRUCK = "pickup truck"
    PICTURE_FRAME = "picture frame"
    PIG = "pig"
    PILLOW = "pillow"
    PINEAPPLE = "pineapple"
    PIZZA = "pizza"
    PLIERS = "pliers"
    POLICE_CAR = "police car"
    POND = "pond"
    POOL = "pool"
    POPSICLE = "popsicle"
    POSTCARD = "postcard"
    POTATO = "potato"
    POWER_OUTLET = "power outlet"
    PURSE = "purse"
    RABBIT = "rabbit"
    RACCOON = "raccoon"
    RADIO = "radio"
    RAIN = "rain"
    RAINBOW = "rainbow"
    RAKE = "rake"
    REMOTE_CONTROL = "remote control"
    RHINOCEROS = "rhinoceros"
    RIFLE = "rifle"
    RIVER = "river"
    ROLLER_COASTER = "roller coaster"
    ROLLERSKATES = "rollerskates"
    SAILBOAT = "sailboat"
    SANDWICH = "sandwich"
    SAW = "saw"
    SAXOPHONE = "saxophone"
    SCHOOL_BUS = "school bus"
    SCISSORS = "scissors"
    SCORPION = "scorpion"
    SCREWDRIVER = "screwdriver"
    SEA_TURTLE = "sea turtle"
    SEE_SAW = "see saw"
    SHARK = "shark"
    SHEEP = "sheep"
    SHOE = "shoe"
    SHORTS = "shorts"
    SHOVEL = "shovel"
    SINK = "sink"
    SKATEBOARD = "skateboard"
    SKULL = "skull"
    SKYSCRAPER = "skyscraper"
    SLEEPING_BAG = "sleeping bag"
    SMILEY_FACE = "smiley face"
    SNAIL = "snail"
    SNAKE = "snake"
    SNORKEL = "snorkel"
    SNOWFLAKE = "snowflake"
    SNOWMAN = "snowman"
    SOCCER_BALL = "soccer ball"
    SOCK = "sock"
    SPEEDBOAT = "speedboat"
    SPIDER = "spider"
    SPOON = "spoon"
    SPREADSHEET = "spreadsheet"
    SQUARE = "square"
    SQUIGGLE = "squiggle"
    SQUIRREL = "squirrel"
    STAIRS = "stairs"
    STAR = "star"
    STEAK = "steak"
    STEREO = "stereo"
    STETHOSCOPE = "stethoscope"
    STITCHES = "stitches"
    STOP_SIGN = "stop sign"
    STOVE = "stove"
    STRAWBERRY = "strawberry"
    STREETLIGHT = "streetlight"
    STRING_BEAN = "string bean"
    SUBMARINE = "submarine"
    SUITCASE = "suitcase"
    SUN = "sun"
    SWAN = "swan"
    SWEATER = "sweater"
    SWING_SET = "swing set"
    SWORD = "sword"
    SYRINGE = "syringe"
    TABLE = "table"
    TEAPOT = "teapot"
    TEDDY_BEAR = "teddy-bear"
    TELEPHONE = "telephone"
    TELEVISION = "television"
    TENNIS_RACQUET = "tennis racquet"
    TENT = "tent"
    THE_EIFFEL_TOWER = "The Eiffel Tower"
    THE_GREAT_WALL_OF_CHINA = "The Great Wall of China"
    THE_MONA_LISA = "The Mona Lisa"
    TIGER = "tiger"
    TOASTER = "toaster"
    TOE = "toe"
    TOILET = "toilet"
    TOOTH = "tooth"
    TOOTHBRUSH = "toothbrush"
    TOOTHPASTE = "toothpaste"
    TORNADO = "tornado"
    TRACTOR = "tractor"
    TRAFFIC_LIGHT = "traffic light"
    TRAIN = "train"
    TREE = "tree"
    TRIANGLE = "triangle"
    TROMBONE = "trombone"
    TRUCK = "truck"
    TRUMPET = "trumpet"
    T_SHIRT = "t-shirt"
    UMBRELLA = "umbrella"
    UNDERWEAR = "underwear"
    VAN = "van"
    VASE = "vase"
    VIOLIN = "violin"
    WASHING_MACHINE = "washing machine"
    WATERMELON = "watermelon"
    WATERSLIDE = "waterslide"
    WHALE = "whale"
    WHEEL = "wheel"
    WINDMILL = "windmill"
    WINE_BOTTLE = "wine bottle"
    WINE_GLASS = "wine glass"
    WRISTWATCH = "wristwatch"
    YOGA = "yoga"
    ZEBRA = "zebra"
    ZIGZAG = "zigzag"

    @property
    def query(self):
        return self.value.replace(" ", "%20")

    def __str__(self):
        return self.value

def get_bounds(data, factor=10):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i][0]) / factor
        y = float(data[i][1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)

def get_factor(strokes, max_dim=224, min_factor=1, max_factor=10):
    """Binary search to find the best scaling factor to fit the strokes within the max dimension."""
    
    best_factor = max_factor
    while min_factor < max_factor - 0.01:
        mid_factor = (min_factor + max_factor) / 2
        min_x, max_x, min_y, max_y = get_bounds(np.array(strokes), factor=mid_factor)
        width = max_x - min_x
        height = max_y - min_y
        
        if width <= max_dim and height <= max_dim:
            best_factor = mid_factor
            max_factor = mid_factor - 0.1  # Try smaller factor to scale up
        else:
            min_factor = mid_factor + 0.1  # Increase factor to scale down

    return best_factor

def draw_strokes(draw, start_x, start_y, factor, strokes, colors):
    current_x, current_y = start_x, start_y
    prev_pen_state = 1
    current_color = random.choice(colors)

    for dx, dy, pen_state in strokes:
        new_x = current_x + dx / factor
        new_y = current_y + dy / factor

        if prev_pen_state == 0:
            draw.line([current_x, current_y, new_x, new_y], fill=current_color, width=random.randint(4, 8))

        if pen_state == 1:
            #color_index = (color_index + 1) % len(colors)
            current_color = random.choice(colors)

        current_x, current_y = new_x, new_y
        prev_pen_state = pen_state

#def draw_canvas(strokes, colors, factor=10, optimized_size = 224):
def draw_canvas(strokes, colors, background = 'white', max_dim=224):
    """Draw strokes on a canvas with random padding, random starting positions, and controlled dimensions using get_factor."""
    
    # Use the get_factor function to determine the appropriate scaling factor
    best_factor = get_factor(strokes, max_dim=max_dim)
    """Draw strokes on a canvas with random padding and random starting positions, then return the image."""
    min_x, max_x, min_y, max_y = get_bounds(np.array(strokes), factor=best_factor)

    # Calculate the width and height of the strokes
    width_stroke = max_x - min_x
    height_stroke = max_y - min_y

    # Generate random padding values within the specified range
    padding_x = random.uniform(width_stroke / 6, width_stroke / 5)
    padding_y = random.uniform(height_stroke / 6, height_stroke / 5)

    # Calculate canvas size including random padding
    width = int(width_stroke + 2 * padding_x)
    height = int(height_stroke + 2 * padding_y)

    # Generate random offsets for the starting positions within specified range
    random_offset_x = random.uniform(padding_x / 2, 1.5 * padding_x)
    random_offset_y = random.uniform(padding_y / 2, 1.5 * padding_y)

    # Calculate starting positions with random offsets
    start_x = -min_x + random_offset_x
    start_y = -min_y + random_offset_y

    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    draw_strokes(draw, start_x, start_y, factor=best_factor, strokes=strokes, colors=colors)
    return image

def merge_canvas(image1, image2, mode='horizontal', background = 'white'):
    """Merge two images either horizontally or vertically."""
    if mode == 'horizontal':
        merged_width = image1.width + image2.width
        merged_height = max(image1.height, image2.height)
        merged_image = Image.new("RGB", (merged_width, merged_height))
        merged_image.paste(image1, (0, 0))
        merged_image.paste(image2, (image1.width, 0))
    elif mode == 'vertical':
        merged_width = max(image1.width, image2.width)
        merged_height = image1.height + image2.height
        merged_image = Image.new("RGB", (merged_width, merged_height), background)
        merged_image.paste(image1, (0, 0))
        merged_image.paste(image2, (0, image1.height))
    return merged_image

def choose_more_square_like(image1, image2, background = 'white'):
    """Choose the more square-like image between the horizontal and vertical combinations."""
    # Combine horizontally
    merged_width_horizontal = image1.width + image2.width
    merged_height_horizontal = max(image1.height, image2.height)
    aspect_ratio_horizontal = merged_width_horizontal / merged_height_horizontal

    # Combine vertically
    merged_width_vertical = max(image1.width, image2.width)
    merged_height_vertical = image1.height + image2.height
    aspect_ratio_vertical = merged_width_vertical / merged_height_vertical

    # Calculate the difference from 1 (perfect square aspect ratio)
    horizontal_diff = abs(aspect_ratio_horizontal - 1)
    vertical_diff = abs(aspect_ratio_vertical - 1)

    # Return the more square-like image
    if horizontal_diff < vertical_diff:
        # Merge horizontally
        merged_image = Image.new("RGB", (merged_width_horizontal, merged_height_horizontal), background)
        merged_image.paste(image1, (0, 0))
        merged_image.paste(image2, (image1.width, 0))
        return merged_image
    else:
        # Merge vertically
        merged_image = Image.new("RGB", (merged_width_vertical, merged_height_vertical), background)
        merged_image.paste(image1, (0, 0))
        merged_image.paste(image2, (0, image1.height))
        return merged_image