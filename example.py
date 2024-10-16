
from recommendation_system import GeneralRecommendationSystem

custom_data = [
    {"id": "101", "name": "Apple iPhone 13",
        "tags": "electronics smartphone mobile iOS A15 Bionic camera display"},
    {"id": "102", "name": "Samsung Galaxy S21",
        "tags": "electronics smartphone mobile Android Snapdragon AMOLED camera"},
    {"id": "103", "name": "Google Pixel 6",
        "tags": "electronics smartphone Android mobile Tensor camera AI"},
    {"id": "104", "name": "Sony WH-1000XM4",
        "tags": "electronics headphones wireless noise-cancelling over-ear audio"},
    {"id": "105", "name": "Bose QuietComfort 35",
        "tags": "electronics headphones wireless noise-cancelling comfortable audio"},
    {"id": "106", "name": "Apple MacBook Pro",
        "tags": "electronics laptop macOS Apple M1 retina lightweight"},
    {"id": "107", "name": "Dell XPS 13",
        "tags": "electronics laptop Windows Intel Core i7 portable lightweight"},
    {"id": "108", "name": "HP Spectre x360",
        "tags": "electronics laptop Windows convertible touchscreen stylus"},
    {"id": "109", "name": "Microsoft Surface Laptop 4",
        "tags": "electronics laptop Windows touchscreen Intel AMD portable"},
    {"id": "110", "name": "Asus ROG Strix G15",
        "tags": "electronics laptop gaming AMD Ryzen NVIDIA RGB keyboard"},
    {"id": "111", "name": "Sony PlayStation 5",
        "tags": "electronics gaming console 4K next-gen DualSense"},
    {"id": "112", "name": "Microsoft Xbox Series X",
        "tags": "electronics gaming console 4K next-gen powerful Game Pass"},
    {"id": "113", "name": "Nintendo Switch",
        "tags": "electronics gaming console portable handheld family-friendly"},
    {"id": "114", "name": "JBL Charge 5",
        "tags": "electronics speaker portable wireless Bluetooth waterproof high-bass"},
    {"id": "115", "name": "Sonos One",
        "tags": "electronics speaker smart wireless multi-room voice control"},
    {"id": "116", "name": "Apple AirPods Pro",
        "tags": "electronics headphones wireless noise-cancelling in-ear spatial audio"},
    {"id": "117", "name": "Samsung Galaxy Buds Pro",
        "tags": "electronics headphones wireless noise-cancelling water-resistant"},
    {"id": "118", "name": "Fitbit Charge 5",
        "tags": "electronics fitness tracker health sleep monitor waterproof"},
    {"id": "119", "name": "Garmin Forerunner 945",
        "tags": "electronics smartwatch GPS multisport fitness heart rate monitor"},
    {"id": "120", "name": "Apple Watch Series 7",
        "tags": "electronics smartwatch health fitness ECG heart rate monitor"},
    {"id": "121", "name": "Razer Blade 15",
        "tags": "electronics laptop gaming Intel Core i7 NVIDIA high-refresh RGB"},
    {"id": "122", "name": "Lenovo ThinkPad X1 Carbon",
        "tags": "electronics laptop Windows Intel Core i7 business durable"},
    {"id": "123", "name": "Google Nest Hub",
        "tags": "electronics smart display home control voice assistant"},
    {"id": "124", "name": "Amazon Echo Show 8",
        "tags": "electronics smart display Alexa voice control home automation"},
    {"id": "125", "name": "Oculus Quest 2",
        "tags": "electronics VR headset standalone wireless gaming motion tracking"},
    {"id": "126", "name": "Logitech MX Master 3",
        "tags": "electronics mouse ergonomic wireless precision customizable"},
    {"id": "127", "name": "Anker PowerCore 10000",
        "tags": "electronics portable charger power bank fast charging"},
    {"id": "128", "name": "DJI Mavic Air 2",
        "tags": "electronics drone 4K camera aerial photography long-flight obstacle avoidance"},
    {"id": "129", "name": "GoPro HERO10 Black",
        "tags": "electronics action camera 4K waterproof rugged stabilization"},
    {"id": "130", "name": "Canon EOS R5",
        "tags": "electronics camera mirrorless 8K video full-frame professional"},
    {"id": "131", "name": "Samsung Galaxy Tab S7",
        "tags": "electronics tablet Android S Pen productivity display"},
    {"id": "132", "name": "Apple iPad Pro",
        "tags": "electronics tablet iOS Apple Pencil productivity retina display"},
    {"id": "133", "name": "Lenovo Yoga Smart Tab",
        "tags": "electronics tablet Android productivity multi-mode display"},
    {"id": "134", "name": "Sony Alpha A7 III",
        "tags": "electronics camera mirrorless full-frame 4K video"},
    {"id": "135", "name": "Nikon Z6 II",
        "tags": "electronics camera mirrorless full-frame video dual card slots"},
    {"id": "136", "name": "Fujifilm X-T4",
        "tags": "electronics camera mirrorless APS-C stabilization 4K video"},
    {"id": "137", "name": "Rode NT-USB Microphone",
        "tags": "electronics microphone USB studio podcasting"},
    {"id": "138", "name": "Shure SM7B",
        "tags": "electronics microphone XLR studio podcasting broadcast"},
    {"id": "139", "name": "Elgato Stream Deck",
        "tags": "electronics streaming controller customizable buttons"},
    {"id": "140", "name": "Blue Yeti USB Microphone",
        "tags": "electronics microphone USB podcasting streaming"},
    {"id": "141", "name": "Corsair K95 RGB Platinum",
        "tags": "electronics keyboard gaming mechanical RGB"},
    {"id": "142", "name": "SteelSeries Apex Pro",
        "tags": "electronics keyboard gaming mechanical RGB customizable"},
    {"id": "143", "name": "HyperX Alloy FPS Pro",
        "tags": "electronics keyboard gaming mechanical compact"},
    {"id": "144", "name": "Asus TUF Gaming A15",
        "tags": "electronics laptop gaming AMD Ryzen RGB keyboard"},
    {"id": "145", "name": "MSI GS66 Stealth",
        "tags": "electronics laptop gaming Intel Core i7 NVIDIA RTX thin"},
    {"id": "146", "name": "HP Omen 15",
        "tags": "electronics laptop gaming AMD Ryzen NVIDIA RGB keyboard"},
    {"id": "147", "name": "LG Gram 17",
        "tags": "electronics laptop Windows Intel Core i7 lightweight"},
    {"id": "148", "name": "Dell Alienware m15 R4",
        "tags": "electronics laptop gaming Intel Core i7 NVIDIA RTX RGB"},
    {"id": "149", "name": "Roku Streaming Stick+",
        "tags": "electronics streaming media 4K HDR portable voice control"},
    {"id": "150", "name": "Chromecast with Google TV",
        "tags": "electronics streaming media 4K HDR voice control Google Assistant"},
]


# Initialize the recommendation system for 'tags' column with 'id' and 'name'
rec_system = GeneralRecommendationSystem(
    feature_column="tags", id_column="id", name_column="name")

# Update the dataset and similarity matrix with custom data or create a new dataset if it doesn't exist
rec_system.update_data_and_similarity(
    custom_data, file_prefix="recommendations")

# Recommend items based on a selected product's name
recommendations = rec_system.recommend(
    "Apple macbook pro", file_prefix="recommendations", top_n=5)

# Output the recommended items
print("Recommended Items:", recommendations)

# Output:
'''
Recommended Items: [
    {'id': '147', 'name': 'lg gram 17'}, 
    {'id': '107', 'name': 'dell xps 13'}, 
    {'id': '132', 'name': 'apple ipad pro'},
    {'id': '108', 'name': 'hp spectre x360'},
    {'id': '109', 'name': 'microsoft surface laptop 4'}
    ]
'''
