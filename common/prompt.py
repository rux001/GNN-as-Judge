
CORA_DIRECT = """Question: Which of the following sub-categories of AI does this paper belong to? Here are the 7 categories: Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods. Reply only one category that you think this paper might belong to. Only reply the category phrase without any other explanation words.\n\nAnswer: """
PUBMED_DIRECT = """Question: Which of the following topic does this scientific publication talk about? Here are the 3 categories: Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
CITESEER_DIRECT = """Question: Which of the following theme does this paper belong to? Here are the 6 categories: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence). Reply only one category that you think this paper might belong to. Only reply the category full name I give you without any other words.\n\nAnswer: """
WIKICS_DIRECT = """Question: Which of the following branch of Computer science does this Wikipedia-based dataset belong to? Here are the 10 categories: Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, Internet Protocols, Computer File Systems, Distributed Computing Architecture, Web Technology, Programming Language Topics. Reply only one category that you think this paper might belong to. Only reply the category full name without any other words.\n\nAnswer: """
ARXIV_DIRECT = """Question: Which of the following 
        arXiv CS sub-categories does this dataset belong to? Here are the 40 categories: 
        'arxiv cs na', 'arxiv cs mm', 'arxiv cs lo', 'arxiv cs cy', 'arxiv cs cr', 
        'arxiv cs dc', 'arxiv cs hc', 'arxiv cs ce', 'arxiv cs ni', 'arxiv cs cc',
        'arxiv cs ai', 'arxiv cs ma', 'arxiv cs gl', 'arxiv cs ne', 'arxiv cs sc', 
        'arxiv cs ar', 'arxiv cs cv', 'arxiv cs gr', 'arxiv cs et', 'arxiv cs sy', 
        'arxiv cs cg', 'arxiv cs oh', 'arxiv cs pl', 'arxiv cs se', 'arxiv cs lg', 
        'arxiv cs sd', 'arxiv cs si', 'arxiv cs ro', 'arxiv cs it', 'arxiv cs pf', 
        'arxiv cs cl', 'arxiv cs ir', 'arxiv cs ms', 'arxiv cs fl', 'arxiv cs ds', 
        'arxiv cs os', 'arxiv cs gt', 'arxiv cs db', 'arxiv cs dl', 'arxiv cs dm'. 
        Use the words in this part to answer me, not the explanation part bellow.

        Here are the explanation of each category:
        'arxiv cs ai (Artificial Intelligence)',
        'arxiv cs ar (Hardware Architecture)',
        'arxiv cs cc (Computational Complexity)',
        'arxiv cs ce (Computational Engineering, Finance, and Science)',
        'arxiv cs cg (Computational Geometry)',
        'arxiv cs cl (Computation and Language)',
        'arxiv cs cr (Cryptography and Security)',
        'arxiv cs cv (Computer Vision and Pattern Recognition)',
        'arxiv cs cy (Computers and Society)',
        'arxiv cs db (Databases)',
        'arxiv cs dc (Distributed, Parallel, and Cluster Computing)',
        'arxiv cs dl (Digital Libraries)',
        'arxiv cs dm (Discrete Mathematics)',
        'arxiv cs ds (Data Structures and Algorithms)',
        'arxiv cs et (Emerging Technologies)',
        'arxiv cs fl (Formal Languages and Automata Theory)',
        'arxiv cs gl (General Literature)',
        'arxiv cs gr (Graphics)',
        'arxiv cs gt (Computer Science and Game Theory)',
        'arxiv cs hc (Human-Computer Interaction)',
        'arxiv cs ir (Information Retrieval)',
        'arxiv cs it (Information Theory)',
        'arxiv cs lg (Machine Learning)',
        'arxiv cs lo (Logic in Computer Science)',
        'arxiv cs ma (Multiagent Systems)',
        'arxiv cs mm (Multimedia)',
        'arxiv cs ms (Mathematical Software)',
        'arxiv cs na (Numerical Analysis)',
        'arxiv cs ne (Neural and Evolutionary Computing)',
        'arxiv cs ni (Networking and Internet Architecture)',
        'arxiv cs oh (Other Computer Science)',
        'arxiv cs os (Operating Systems)',
        'arxiv cs pf (Performance)',
        'arxiv cs pl (Programming Languages)',
        'arxiv cs ro (Robotics)',
        'arxiv cs sc (Symbolic Computation)',
        'arxiv cs sd (Sound)',
        'arxiv cs se (Software Engineering)',
        'arxiv cs si (Social and Information Networks)',
        'arxiv cs sy (Systems and Control)'
        Reply only one category that you think this paper might belong to. 
        Only reply the category name (not the explanation) I given without any other words, please don't use your own words.Be careful, only use the name of the category I give you, not the explanation part or any other words.\n\nAnswer:"""
INSTAGRAM_DIRECT = """Question: Which of the following categories does this instagram user belong to? Here are the 2 categories: Normal Users, Commercial Users. Reply only one category that you think this user might belong to. Only reply the category name I give of the category: Normal Users, Commercial Users, without any other words.\n\nAnswer: """
REDDIT_DIRECT = """Question: Which of the following categories does this reddit user belong to? Here are the 2 categories: Normal Users, Popular Users. Popular Users' posted content are often more attractive. Reply only one category that you think this user might belong to. Only reply the category name I give of the category: Normal Users, Popular Users, without any other words.\n\nAnswer: """
PHOTO_DIRECT = "Which of the following categories does this photo item belong to? Here are the 12 categories: Video Surveillance, Accessories, Binoculars & Scopes, Video, Lighting & Studio, Bags & Cases, Tripods & Monopods, Flashes, Digital Cameras, Film Photography, Lenses, Underwater Photography. Reply only one category that you think this item might belong to. Only reply the category name I give of the category without any other words.\n\nAnswer: """
PRODUCT_DIRECT = "Which of the following categories does this product belong to? There are a total of 47 categories, including Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, Baby Products, NAN, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Kindle Apps, Furniture. Reply only one category that you think this product might belong to. Only reply the category name I give of the category without any other words and numbers.\n\nAnswer: "
WISCONSIN_DIRECT = "Classify this school webpage into one of the following 5 categories:\n\nstudent, faculty, staff, course, project\n\nRespond with only the exact category name from the list above.\n\nAnswer: "
CORNELL_DIRECT = "Classify this school webpage into one of the following 5 categories:\n\nstudent, faculty, staff, course, project\n\nRespond with only the exact category name from the list above.\n\nAnswer: "

DIRECT_PROMPTS = {
    "cora": CORA_DIRECT,
    "pubmed": PUBMED_DIRECT,
    "citeseer": CITESEER_DIRECT,
    "wikics": WIKICS_DIRECT,
    "arxiv": ARXIV_DIRECT,
    "instagram": INSTAGRAM_DIRECT,
    "reddit": REDDIT_DIRECT,
    "photo": PHOTO_DIRECT,
    "ogbn-products": PRODUCT_DIRECT,
    "cornell": CORNELL_DIRECT,
    "wisconsin": WISCONSIN_DIRECT,
    "ogbn-products_subset": PRODUCT_DIRECT,
}

