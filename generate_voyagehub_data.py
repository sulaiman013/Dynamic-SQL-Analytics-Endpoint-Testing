"""
VoyageHub Synthetic Dataset Generator
======================================
Generates a realistic, production-quality flat dataset for a travel super-app.
Output: Snappy-compressed Parquet files, each under 95MB.

Usage:
    python generate_voyagehub_data.py --mode smoke          # 1.5M rows
    python generate_voyagehub_data.py --mode prod           # 60M rows
    python generate_voyagehub_data.py --mode prod --hash-emails  # privacy mode
"""

import argparse
import gc
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Constants: Geography
# ---------------------------------------------------------------------------

USER_COUNTRY_LIST = [
    "USA", "UK", "India", "Australia", "Germany",
    "Singapore", "UAE", "Canada", "Japan", "France",
    "South Korea", "Netherlands", "Brazil", "Thailand", "Malaysia",
]
USER_COUNTRY_WEIGHTS = [
    0.25, 0.10, 0.10, 0.08, 0.07,
    0.05, 0.05, 0.05, 0.05, 0.03,
    0.03, 0.03, 0.03, 0.04, 0.04,
]

DESTINATION_COUNTRY_LIST = [
    "Indonesia", "Thailand", "Malaysia", "Japan", "Singapore",
    "Maldives", "Vietnam", "Philippines", "South Korea", "Sri Lanka",
    "India", "UAE", "Australia", "Turkey", "Nepal",
    "Cambodia", "Myanmar", "Laos", "New Zealand", "Fiji",
]
DESTINATION_COUNTRY_WEIGHTS = [
    0.15, 0.12, 0.10, 0.10, 0.08,
    0.05, 0.05, 0.05, 0.05, 0.03,
    0.03, 0.03, 0.02, 0.02, 0.02,
    0.02, 0.02, 0.02, 0.02, 0.02,
]

COUNTRY_CITY_MAP = {
    "USA": ["New York", "Los Angeles", "Chicago", "Houston", "Miami", "San Francisco", "Seattle", "Boston", "Denver", "Atlanta"],
    "UK": ["London", "Manchester", "Birmingham", "Edinburgh", "Bristol", "Liverpool", "Leeds", "Glasgow"],
    "India": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "Pune", "Ahmedabad", "Jaipur"],
    "Australia": ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Gold Coast", "Canberra"],
    "Germany": ["Berlin", "Munich", "Frankfurt", "Hamburg", "Cologne", "Stuttgart", "Dusseldorf"],
    "Singapore": ["Singapore"],
    "UAE": ["Dubai", "Abu Dhabi", "Sharjah"],
    "Canada": ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa", "Edmonton"],
    "Japan": ["Tokyo", "Osaka", "Kyoto", "Yokohama", "Nagoya", "Sapporo", "Fukuoka", "Kobe"],
    "France": ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Bordeaux"],
    "South Korea": ["Seoul", "Busan", "Incheon", "Daegu", "Jeju City"],
    "Netherlands": ["Amsterdam", "Rotterdam", "The Hague", "Utrecht"],
    "Brazil": ["Sao Paulo", "Rio de Janeiro", "Brasilia", "Salvador", "Fortaleza"],
    "Thailand": ["Bangkok", "Chiang Mai", "Pattaya", "Phuket", "Krabi", "Hua Hin"],
    "Malaysia": ["Kuala Lumpur", "Penang", "Johor Bahru", "Kota Kinabalu", "Langkawi", "Malacca"],
    "Indonesia": ["Bali", "Jakarta", "Yogyakarta", "Surabaya", "Lombok", "Bandung", "Medan"],
    "Maldives": ["Male", "Hulhumale", "Addu City", "Fuvahmulah"],
    "Vietnam": ["Ho Chi Minh City", "Hanoi", "Da Nang", "Hoi An", "Nha Trang", "Ha Long"],
    "Philippines": ["Manila", "Cebu", "Boracay", "Palawan", "Davao", "Baguio"],
    "Sri Lanka": ["Colombo", "Kandy", "Galle", "Ella", "Sigiriya"],
    "Turkey": ["Istanbul", "Antalya", "Cappadocia", "Bodrum", "Izmir"],
    "Nepal": ["Kathmandu", "Pokhara", "Lumbini", "Chitwan"],
    "Cambodia": ["Phnom Penh", "Siem Reap", "Sihanoukville"],
    "Myanmar": ["Yangon", "Mandalay", "Bagan", "Inle Lake"],
    "Laos": ["Vientiane", "Luang Prabang", "Vang Vieng"],
    "New Zealand": ["Auckland", "Queenstown", "Wellington", "Christchurch", "Rotorua"],
    "Fiji": ["Nadi", "Suva", "Denarau"],
}

# UTC offsets (hours) for timestamp generation
COUNTRY_UTC_OFFSET = {
    "USA": -5, "UK": 0, "India": 5.5, "Australia": 10, "Germany": 1,
    "Singapore": 8, "UAE": 4, "Canada": -5, "Japan": 9, "France": 1,
    "South Korea": 9, "Netherlands": 1, "Brazil": -3, "Thailand": 7, "Malaysia": 8,
    "Indonesia": 7, "Maldives": 5, "Vietnam": 7, "Philippines": 8, "Sri Lanka": 5.5,
    "Turkey": 3, "Nepal": 5.75, "Cambodia": 7, "Myanmar": 6.5, "Laos": 7,
    "New Zealand": 12, "Fiji": 12,
}

# Tax rates by destination country (percentage as decimal)
COUNTRY_TAX_RATES = {
    "Indonesia": 0.11, "Thailand": 0.07, "Malaysia": 0.06, "Japan": 0.10, "Singapore": 0.09,
    "Maldives": 0.16, "Vietnam": 0.10, "Philippines": 0.12, "South Korea": 0.10, "Sri Lanka": 0.15,
    "India": 0.18, "UAE": 0.05, "Australia": 0.10, "Turkey": 0.18, "Nepal": 0.13,
    "Cambodia": 0.10, "Myanmar": 0.05, "Laos": 0.10, "New Zealand": 0.15, "Fiji": 0.09,
}

# Default currency by user country
CURRENCY_LIST = ["USD", "EUR", "GBP", "SGD", "AUD", "AED", "INR", "JPY", "CAD", "KRW", "BRL", "THB", "MYR"]
CURRENCY_WEIGHTS = [0.40, 0.10, 0.10, 0.08, 0.07, 0.05, 0.05, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01]

# ---------------------------------------------------------------------------
# Constants: Booking
# ---------------------------------------------------------------------------

BOOKING_TYPES = ["flight", "hotel", "ride"]
BOOKING_TYPE_WEIGHTS = [0.40, 0.35, 0.25]

BOOKING_STATUS_LIST = ["completed", "cancelled", "refunded", "pending", "no_show", "in_progress"]
BOOKING_STATUS_WEIGHTS = [0.72, 0.12, 0.06, 0.05, 0.03, 0.02]

MEMBERSHIP_TIERS = ["Bronze", "Silver", "Gold", "Platinum", "Diamond"]
MEMBERSHIP_WEIGHTS = [0.50, 0.25, 0.15, 0.08, 0.02]

PAYMENT_METHODS = ["credit_card", "debit_card", "digital_wallet", "bank_transfer", "crypto", "buy_now_pay_later"]
PAYMENT_WEIGHTS = [0.40, 0.20, 0.20, 0.10, 0.05, 0.05]

PLATFORMS = ["mobile_app", "web", "partner_api", "call_center"]
PLATFORM_WEIGHTS = [0.55, 0.30, 0.10, 0.05]

DEVICE_OS_LIST = ["iOS", "Android", "Windows", "macOS", "Linux"]
DEVICE_OS_WEIGHTS = [0.36, 0.41, 0.12, 0.08, 0.03]

APP_VERSIONS = [
    "3.8.0", "3.9.1", "4.0.0", "4.0.2", "4.1.0",
    "4.2.1", "4.3.0", "4.4.0", "4.5.1", "5.0.0",
    "5.0.1", "5.1.0", "5.1.2", "5.2.0", "5.3.0",
]
APP_VERSION_WEIGHTS = [
    0.02, 0.02, 0.03, 0.03, 0.04,
    0.05, 0.06, 0.07, 0.08, 0.10,
    0.10, 0.10, 0.10, 0.10, 0.10,
]

CANCELLATION_REASONS = [
    "change_of_plans", "found_better_deal", "emergency",
    "weather", "duplicate_booking", "price_too_high",
]

PROMO_CODES = [
    "SUMMER2024", "WELCOME10", "FLASH50", "LOYALTY20", "NEWYEAR25",
    "SPRING15", "HOLIDAY30", "WEEKEND10", "FIRSTRIDE", "VOYAGER50",
    "EARLYBIRD", "GETAWAY20", "ADVENTURE", "EXPLORE15", "SUNSET25",
    "TROPICAL", "WANDER10", "BLISS2024", "JOURNEY30", "PARADISE",
]

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "voyagehub.com"]
EMAIL_DOMAIN_WEIGHTS = [0.40, 0.20, 0.20, 0.10, 0.10]

# Seasonal booking weights by month (1-indexed). Peaks: summer, holidays, spring break
MONTH_WEIGHTS = [
    0.07, 0.07, 0.09, 0.08, 0.08,
    0.10, 0.11, 0.11, 0.08, 0.07,
    0.06, 0.08,
]

# ---------------------------------------------------------------------------
# Constants: Names (diverse pool)
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    # Western
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Lisa", "Daniel", "Nancy",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
    "Kenneth", "Carol", "Kevin", "Amanda", "Brian", "Dorothy", "George", "Melissa",
    "Timothy", "Deborah", "Ronald", "Stephanie", "Edward", "Rebecca", "Jason", "Sharon",
    "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy",
    "Nicholas", "Angela", "Eric", "Shirley", "Jonathan", "Anna", "Stephen", "Brenda",
    "Larry", "Pamela", "Justin", "Emma", "Scott", "Nicole", "Brandon", "Helen",
    # Asian
    "Wei", "Yuki", "Hiroshi", "Sakura", "Min-jun", "Ji-eun", "Takeshi", "Mei",
    "Ravi", "Priya", "Arjun", "Deepa", "Sanjay", "Anita", "Vikram", "Kavitha",
    "Amit", "Sunita", "Raj", "Pooja", "Suresh", "Lakshmi", "Arun", "Divya",
    "Chen", "Li", "Wang", "Zhang", "Liu", "Huang", "Lin", "Yang",
    "Akira", "Naomi", "Kenji", "Yumi", "Daichi", "Hana", "Haruto", "Aoi",
    "Jian", "Xin", "Feng", "Ying", "Tao", "Lan", "Ming", "Hui",
    "Soo-jin", "Hyun", "Dong-won", "Eun-ji", "Seung", "Yoo-jin", "Tae-ho", "Da-hye",
    "Anh", "Linh", "Duc", "Trang", "Minh", "Hoa", "Quang", "Mai",
    "Nattapong", "Siriporn", "Somchai", "Ploy", "Arthit", "Nong", "Chai", "Fah",
    "Rizal", "Siti", "Ahmad", "Nurhaliza", "Budi", "Dewi", "Adi", "Putri",
    # Middle Eastern
    "Mohammed", "Fatima", "Ahmed", "Aisha", "Ali", "Mariam", "Omar", "Sara",
    "Hassan", "Layla", "Youssef", "Noura", "Khalid", "Huda", "Ibrahim", "Amina",
    "Tariq", "Zainab", "Karim", "Dina", "Nasser", "Reem", "Faisal", "Mona",
    "Saeed", "Hanan", "Rashid", "Salma", "Hamad", "Lina", "Sultan", "Dana",
    # Latin American
    "Carlos", "Maria", "Juan", "Ana", "Luis", "Lucia", "Diego", "Valentina",
    "Pedro", "Camila", "Fernando", "Sofia", "Miguel", "Isabella", "Pablo", "Gabriela",
    "Alejandro", "Paula", "Ricardo", "Daniela", "Andres", "Natalia", "Sebastian", "Carolina",
    "Felipe", "Adriana", "Javier", "Monica", "Roberto", "Elena", "Eduardo", "Rosa",
]

LAST_NAMES = [
    # Western
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
    "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
    "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
    "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
    "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy",
    # Asian
    "Kim", "Park", "Choi", "Jung", "Kang", "Cho", "Yoon", "Jang",
    "Lim", "Han", "Oh", "Seo", "Shin", "Kwon", "Hwang", "Ahn",
    "Tanaka", "Suzuki", "Watanabe", "Sato", "Takahashi", "Ito", "Yamamoto", "Nakamura",
    "Kobayashi", "Kato", "Yoshida", "Yamada", "Sasaki", "Inoue", "Matsumoto", "Kimura",
    "Patel", "Sharma", "Singh", "Kumar", "Gupta", "Joshi", "Shah", "Reddy",
    "Nair", "Menon", "Rao", "Desai", "Bhatt", "Chopra", "Kapoor", "Malhotra",
    "Wong", "Chan", "Tan", "Lau", "Cheung", "Cheng", "Leung", "Ho",
    "Ng", "Wu", "Chua", "Goh", "Teo", "Ong", "Koh", "Yap",
    # Middle Eastern
    "Al-Rashid", "Al-Farsi", "Al-Mansour", "Al-Sayed", "Al-Hamad", "Al-Nasser",
    "El-Amin", "El-Khoury", "Haddad", "Khoury", "Nasser", "Saleh",
    "Hashemi", "Hosseini", "Mohammadi", "Rezaei", "Rahimi", "Karimi",
    # Latin
    "Silva", "Santos", "Oliveira", "Souza", "Lima", "Costa",
    "Ferreira", "Almeida", "Pereira", "Carvalho", "Rocha", "Ribeiro",
    "Fernandez", "Alvarez", "Romero", "Vargas", "Moreno", "Castillo",
]

REVIEW_TEXTS = [
    "Great experience!", "Highly recommended.", "Will book again.",
    "Excellent service, very satisfied.", "Smooth booking process.",
    "Good value for money.", "Amazing hotel, loved the view.",
    "Flight was on time, comfortable seats.", "Ride was clean and quick.",
    "The driver was very professional.", "Hotel staff were incredibly helpful.",
    "Beautiful destination, exceeded expectations.", "Perfect for families.",
    "Bit pricey but worth it.", "Average experience, nothing special.",
    "Could have been better.", "Delayed flight but staff handled it well.",
    "Room was clean but small.", "Great location, walking distance to everything.",
    "Food at the hotel was outstanding.", "Easy check-in process.",
    "The pool area was fantastic.", "Quick and reliable ride service.",
    "Loved the cultural experience.", "Airport pickup was seamless.",
    "Would not recommend this vendor.", "Overpriced for what you get.",
    "Terrible customer service.", "Room did not match the listing.",
    "Flight cancelled with no notice.", "Refund took too long.",
    "Nice but noisy at night.", "Friendly staff, cozy room.",
    "Spectacular sunset views.", "The spa was world class.",
    "Great trip overall, minor hiccups.", "Driver arrived late but was apologetic.",
    "Booking was easy, experience was great.", "Loved the local food tours.",
    "Clean vehicle, safe driving.", "Hotel gym was well equipped.",
    "Beach access was a huge plus.", "Very comfortable bed.",
    "Wi-Fi was fast and reliable.", "Excellent breakfast buffet.",
    "The concierge went above and beyond.", "A bit far from the city center.",
    "Great for a weekend getaway.", "Memorable trip, will return.",
    "Good enough for a business trip.", "Outstanding rooftop bar.",
]

# ---------------------------------------------------------------------------
# Constants: Vendors
# ---------------------------------------------------------------------------

AIRLINE_NAMES = [
    "SkyAsia Airways", "Pacific Wings", "Oriental Express Air", "Golden Eagle Airlines",
    "Coral Jet", "Horizon Pacific", "Monsoon Air", "Sapphire Airlines",
    "TropicAir", "Azure Sky Airlines", "Summit Air", "Bamboo Airlines",
    "Crystal Air", "Jade Airways", "Zenith Airlines", "Voyage Air",
    "Atlas Sky", "Pinnacle Airlines", "Meridian Air", "Equinox Airways",
    "Dawn Airlines", "Coastal Air", "Nomad Airways", "Solstice Air",
    "Polaris Airlines", "Ember Air", "Falcon Airways", "Eclipse Airlines",
    "Nova Air", "Celestial Airlines",
]

HOTEL_NAMES = [
    "Grand Orchid Resort", "Coral Bay Inn", "Sapphire Suites", "Golden Lotus Hotel",
    "The Emerald Palace", "Sunset Beach Resort", "Azure Waters Hotel", "Pearl Harbor Inn",
    "Bamboo Lodge", "Ivory Tower Hotel", "Lotus Garden Resort", "Starlight Suites",
    "The Crystal Pavilion", "Horizon Bay Resort", "Palm Crown Hotel", "Jade Garden Inn",
    "Tropica Grand", "Moonrise Hotel", "Cascade Lodge", "Serenity Resort",
    "The Pinnacle Hotel", "Riverside Suites", "Mountain View Lodge", "Harbor Lights Inn",
    "The Zen Retreat", "Sandcastle Resort", "Skyline Suites", "Ocean Breeze Hotel",
    "Rainforest Lodge", "Silk Road Inn", "Summit Peak Hotel", "Lagoon Resort",
    "Amber Sands Hotel", "Crescent Moon Inn", "Driftwood Resort", "Terra Firma Hotel",
    "Cloudnine Suites", "Windmill Lodge", "Copperfield Hotel", "Mirage Resort",
]

RIDE_COMPANY_NAMES = [
    "SwiftRide", "CityHop", "QuickCab", "MetroGo", "UrbanDrive",
    "ZoomRide", "EasyMove", "SnapCab", "RideNow", "JetSet Rides",
    "TukTuk Express", "Green Wheels", "FastTrack Rides", "LocalMotion",
    "GoRide", "PeakTransit", "Dash Rides", "OneWay Go", "WheelZone",
    "FleetFoot", "SpeedLink", "RoadRunner", "ProRide", "ComfortCab",
    "SafeRide", "TopGear Transit", "NomadRide", "BlueWheel", "EliteCab",
    "CruiseLine Rides",
]

# ---------------------------------------------------------------------------
# PyArrow Schema
# ---------------------------------------------------------------------------

PARQUET_SCHEMA = pa.schema([
    ("transaction_id", pa.string()),
    ("user_id", pa.string()),
    ("user_name", pa.string()),
    ("user_email", pa.string()),
    ("user_country", pa.string()),
    ("user_membership_tier", pa.string()),
    ("booking_type", pa.string()),
    ("booking_date", pa.date32()),
    ("booking_timestamp", pa.timestamp("us", tz="UTC")),
    ("destination_country", pa.string()),
    ("destination_city", pa.string()),
    ("origin_city", pa.string()),
    ("booking_status", pa.string()),
    ("payment_method", pa.string()),
    ("currency", pa.string()),
    ("base_amount", pa.float64()),
    ("tax_amount", pa.float64()),
    ("discount_amount", pa.float64()),
    ("total_amount", pa.float64()),
    ("promo_code", pa.string()),
    ("platform", pa.string()),
    ("device_os", pa.string()),
    ("app_version", pa.string()),
    ("session_duration_seconds", pa.int32()),
    ("is_repeat_booking", pa.bool_()),
    ("rating", pa.float32()),
    ("review_text", pa.string()),
    ("cancellation_reason", pa.string()),
    ("vendor_id", pa.string()),
    ("vendor_name", pa.string()),
])


# ---------------------------------------------------------------------------
# Pool generators (one-time setup, reused across chunks)
# ---------------------------------------------------------------------------

def generate_user_pool(rng: np.random.Generator, n_users: int = 2_000_000) -> dict:
    """Pre-generate a pool of unique user profiles."""
    print(f"Generating user pool ({n_users:,} users)...")
    start = time.time()

    user_ids = np.array([f"USR-{i:07d}" for i in range(1, n_users + 1)])

    first_idx = rng.integers(0, len(FIRST_NAMES), n_users)
    last_idx = rng.integers(0, len(LAST_NAMES), n_users)
    first_names = np.array(FIRST_NAMES)[first_idx]
    last_names = np.array(LAST_NAMES)[last_idx]
    full_names = np.char.add(np.char.add(first_names, " "), last_names)

    domain_idx = rng.choice(len(EMAIL_DOMAINS), n_users, p=EMAIL_DOMAIN_WEIGHTS)
    domains = np.array(EMAIL_DOMAINS)[domain_idx]
    email_prefixes = np.char.add(
        np.char.lower(first_names),
        np.char.add(np.full(n_users, "."), np.char.lower(last_names)),
    )
    # Add numeric suffix for uniqueness
    suffixes = np.array([str(i) for i in range(n_users)])
    email_prefixes = np.char.add(email_prefixes, suffixes)
    emails = np.char.add(np.char.add(email_prefixes, np.full(n_users, "@")), domains)

    countries = rng.choice(USER_COUNTRY_LIST, n_users, p=USER_COUNTRY_WEIGHTS)
    tiers = rng.choice(MEMBERSHIP_TIERS, n_users, p=MEMBERSHIP_WEIGHTS)

    elapsed = time.time() - start
    print(f"  User pool ready in {elapsed:.1f}s")

    return {
        "user_id": user_ids,
        "user_name": full_names,
        "user_email": emails,
        "user_country": countries,
        "user_membership_tier": tiers,
    }


def generate_vendor_pool(rng: np.random.Generator, n_vendors: int = 5000) -> dict:
    """Pre-generate a pool of vendors (airlines, hotels, ride companies)."""
    print(f"Generating vendor pool ({n_vendors:,} vendors)...")

    vendor_ids = np.array([f"VND-{i:05d}" for i in range(1, n_vendors + 1)])

    # Assign types: ~30% airlines, ~40% hotels, ~30% ride companies
    n_airlines = int(n_vendors * 0.30)
    n_hotels = int(n_vendors * 0.40)
    n_rides = n_vendors - n_airlines - n_hotels

    vendor_types = (
        ["flight"] * n_airlines +
        ["hotel"] * n_hotels +
        ["ride"] * n_rides
    )
    rng.shuffle(vendor_types)
    vendor_types = np.array(vendor_types)

    vendor_names = np.empty(n_vendors, dtype=object)
    for i in range(n_vendors):
        vtype = vendor_types[i]
        if vtype == "flight":
            vendor_names[i] = AIRLINE_NAMES[i % len(AIRLINE_NAMES)]
        elif vtype == "hotel":
            vendor_names[i] = HOTEL_NAMES[i % len(HOTEL_NAMES)]
        else:
            vendor_names[i] = RIDE_COMPANY_NAMES[i % len(RIDE_COMPANY_NAMES)]

    return {
        "vendor_id": vendor_ids,
        "vendor_name": vendor_names,
        "vendor_type": vendor_types,
    }


# ---------------------------------------------------------------------------
# Chunk generation
# ---------------------------------------------------------------------------

def generate_chunk(
    n: int,
    rng: np.random.Generator,
    user_pool: dict,
    vendor_pool: dict,
) -> pd.DataFrame:
    """Generate a single chunk of n rows with all 30 columns."""

    n_users = len(user_pool["user_id"])
    n_vendors = len(vendor_pool["vendor_id"])

    # Step 1: Sample users from pool
    user_indices = rng.integers(0, n_users, n)
    user_id = user_pool["user_id"][user_indices]
    user_name = user_pool["user_name"][user_indices]
    user_email = user_pool["user_email"][user_indices]
    user_country = user_pool["user_country"][user_indices]
    user_membership_tier = user_pool["user_membership_tier"][user_indices]

    # Step 2: Booking type
    booking_type = rng.choice(BOOKING_TYPES, n, p=BOOKING_TYPE_WEIGHTS)

    # Step 3: Booking date (2022-01-01 to 2025-12-31) -- fully vectorized
    years = rng.choice([2022, 2023, 2024, 2025], n)
    months = rng.choice(np.arange(1, 13), n, p=MONTH_WEIGHTS)

    # Vectorized days-in-month calculation
    days_in_month = np.where(
        np.isin(months, [1, 3, 5, 7, 8, 10, 12]), 31,
        np.where(
            np.isin(months, [4, 6, 9, 11]), 30,
            np.where(
                (years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0)), 29, 28
            )
        )
    ).astype(np.int32)

    random_days = (rng.random(n) * days_in_month).astype(np.int32)
    random_days = np.clip(random_days, 0, days_in_month - 1)
    day_of_month = random_days + 1

    # Build dates using pandas vectorized construction (no per-row loop)
    booking_date_ts = pd.to_datetime(
        pd.DataFrame({"year": years, "month": months, "day": day_of_month})
    )
    booking_date = booking_date_ts.dt.date

    # Step 4: Booking timestamp -- vectorized with timezone offset
    hour_weights = np.array([
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05,
        0.07, 0.09, 0.09, 0.07, 0.05, 0.04, 0.04, 0.04,
        0.05, 0.05, 0.06, 0.07, 0.06, 0.04, 0.02, 0.01,
    ])
    hour_weights = hour_weights / hour_weights.sum()
    hours = rng.choice(24, n, p=hour_weights)
    minutes = rng.integers(0, 60, n)
    seconds = rng.integers(0, 60, n)

    # Build local timestamps as numpy datetime64, then subtract UTC offset vectorized
    time_of_day_seconds = hours.astype(np.int64) * 3600 + minutes.astype(np.int64) * 60 + seconds.astype(np.int64)
    local_timestamps = booking_date_ts.values + pd.to_timedelta(time_of_day_seconds, unit="s").values

    # Vectorized UTC offset lookup by country grouping
    utc_offsets_hours = np.zeros(n, dtype=np.float64)
    for country, offset in COUNTRY_UTC_OFFSET.items():
        mask = user_country == country
        utc_offsets_hours[mask] = offset
    utc_offset_ns = (utc_offsets_hours * 3600 * 1e9).astype(np.int64)

    utc_timestamps = local_timestamps - utc_offset_ns.astype("timedelta64[ns]")
    booking_timestamp = pd.DatetimeIndex(utc_timestamps).tz_localize("UTC")

    # Step 5: Geography -- vectorized by grouping on country
    dest_country = rng.choice(DESTINATION_COUNTRY_LIST, n, p=DESTINATION_COUNTRY_WEIGHTS)

    dest_city = np.empty(n, dtype=object)
    origin_city = np.empty(n, dtype=object)

    # Vectorized city assignment: group by country, batch assign
    for country, cities in COUNTRY_CITY_MAP.items():
        cities_arr = np.array(cities)
        # Destination cities
        dc_mask = dest_country == country
        dc_count = dc_mask.sum()
        if dc_count > 0:
            dest_city[dc_mask] = cities_arr[rng.integers(0, len(cities_arr), dc_count)]
        # Origin cities
        oc_mask = user_country == country
        oc_count = oc_mask.sum()
        if oc_count > 0:
            origin_city[oc_mask] = cities_arr[rng.integers(0, len(cities_arr), oc_count)]

    # Step 6: Booking status
    booking_status = rng.choice(BOOKING_STATUS_LIST, n, p=BOOKING_STATUS_WEIGHTS)

    # Step 7: Financial columns
    base_amount = np.empty(n, dtype=np.float64)
    flight_mask = booking_type == "flight"
    hotel_mask = booking_type == "hotel"
    ride_mask = booking_type == "ride"

    n_flights = flight_mask.sum()
    n_hotels = hotel_mask.sum()
    n_rides = ride_mask.sum()

    if n_flights > 0:
        base_amount[flight_mask] = np.clip(
            rng.lognormal(mean=5.8, sigma=0.6, size=n_flights), 80, 2500
        )
    if n_hotels > 0:
        base_amount[hotel_mask] = np.clip(
            rng.lognormal(mean=4.8, sigma=0.7, size=n_hotels), 25, 800
        )
    if n_rides > 0:
        base_amount[ride_mask] = np.clip(
            rng.lognormal(mean=2.8, sigma=0.8, size=n_rides), 3, 150
        )
    base_amount = np.round(base_amount, 2)

    # Tax based on destination country (vectorized lookup)
    _tax_default = 0.10
    _tax_lookup = np.full(n, _tax_default, dtype=np.float64)
    for country, rate in COUNTRY_TAX_RATES.items():
        _tax_lookup[dest_country == country] = rate
    tax_rates = _tax_lookup
    # Add slight variation (+/- 2%)
    tax_rates = tax_rates + rng.uniform(-0.02, 0.02, n)
    tax_rates = np.clip(tax_rates, 0.05, 0.18)
    tax_amount = np.round(base_amount * tax_rates, 2)

    # Discount: 60% get no discount
    has_discount = rng.random(n) >= 0.60
    discount_amount = np.zeros(n, dtype=np.float64)
    n_discounted = has_discount.sum()
    if n_discounted > 0:
        raw_discount = rng.uniform(5, 200, n_discounted)
        max_discount = base_amount[has_discount] * 0.30
        discount_amount[has_discount] = np.minimum(raw_discount, max_discount)
    discount_amount = np.round(discount_amount, 2)

    # Promo code: 70% NULL. When promo exists, discount must be > 0
    promo_code = np.full(n, None, dtype=object)
    has_promo = rng.random(n) >= 0.70
    n_promo = has_promo.sum()
    if n_promo > 0:
        promo_code[has_promo] = rng.choice(PROMO_CODES, n_promo)
        # Ensure discount > 0 when promo exists
        promo_no_discount = has_promo & (discount_amount <= 0)
        n_fix = promo_no_discount.sum()
        if n_fix > 0:
            max_disc = np.maximum(base_amount[promo_no_discount] * 0.15, 6.0)
            max_disc = np.minimum(max_disc, 50.0)
            discount_amount[promo_no_discount] = np.round(
                rng.uniform(5, max_disc), 2
            )

    total_amount = np.round(base_amount + tax_amount - discount_amount, 2)
    total_amount = np.maximum(total_amount, 0)

    # Step 8: Currency
    currency = rng.choice(CURRENCY_LIST, n, p=CURRENCY_WEIGHTS)

    # Step 9: Payment method
    payment_method = rng.choice(PAYMENT_METHODS, n, p=PAYMENT_WEIGHTS)

    # Step 10: Platform and device
    platform = rng.choice(PLATFORMS, n, p=PLATFORM_WEIGHTS)
    device_os = np.empty(n, dtype=object)
    call_center_mask = platform == "call_center"
    non_cc_mask = ~call_center_mask
    n_non_cc = non_cc_mask.sum()
    if n_non_cc > 0:
        device_os[non_cc_mask] = rng.choice(DEVICE_OS_LIST, n_non_cc, p=DEVICE_OS_WEIGHTS)
    device_os[call_center_mask] = None

    app_version = rng.choice(APP_VERSIONS, n, p=APP_VERSION_WEIGHTS)

    # Step 11: Session duration (varies by booking type)
    session_duration = np.empty(n, dtype=np.int32)
    if n_flights > 0:
        session_duration[flight_mask] = np.clip(
            rng.normal(900, 400, n_flights).astype(np.int32), 60, 3600
        )
    if n_hotels > 0:
        session_duration[hotel_mask] = np.clip(
            rng.normal(600, 300, n_hotels).astype(np.int32), 45, 3600
        )
    if n_rides > 0:
        session_duration[ride_mask] = np.clip(
            rng.normal(300, 150, n_rides).astype(np.int32), 30, 1800
        )

    # Step 12: is_repeat_booking
    is_repeat = rng.random(n) < 0.35

    # Step 13: Rating (NULL for 40%, NULL if pending/in_progress, left-skewed mean=4.1)
    rating = np.full(n, np.nan, dtype=np.float32)
    can_rate = ~np.isin(booking_status, ["pending", "in_progress"])
    has_rating_mask = can_rate & (rng.random(n) >= 0.40)
    n_rated = has_rating_mask.sum()
    if n_rated > 0:
        # Left-skewed: use beta distribution scaled to 1-5
        raw_ratings = rng.beta(a=5.0, b=1.8, size=n_rated) * 4 + 1  # range 1-5, skewed high
        rating[has_rating_mask] = np.round(raw_ratings * 2) / 2  # round to 0.5

    # Step 14: Review text (NULL for 80%)
    review_text = np.full(n, None, dtype=object)
    has_review = rng.random(n) >= 0.80
    n_reviews = has_review.sum()
    if n_reviews > 0:
        review_text[has_review] = rng.choice(REVIEW_TEXTS, n_reviews)

    # Step 15: Cancellation reason (NULL unless cancelled/refunded)
    cancellation_reason = np.full(n, None, dtype=object)
    cancel_mask = np.isin(booking_status, ["cancelled", "refunded"])
    n_cancelled = cancel_mask.sum()
    if n_cancelled > 0:
        cancellation_reason[cancel_mask] = rng.choice(CANCELLATION_REASONS, n_cancelled)

    # Step 16: Vendors (matched by booking type)
    vendor_id_col = np.empty(n, dtype=object)
    vendor_name_col = np.empty(n, dtype=object)

    for bt in BOOKING_TYPES:
        bt_mask = booking_type == bt
        bt_count = bt_mask.sum()
        if bt_count == 0:
            continue
        # Find vendors of this type
        vtype_mask = vendor_pool["vendor_type"] == bt
        vtype_indices = np.where(vtype_mask)[0]
        if len(vtype_indices) == 0:
            vtype_indices = np.arange(len(vendor_pool["vendor_id"]))
        chosen = rng.choice(vtype_indices, bt_count)
        vendor_id_col[bt_mask] = vendor_pool["vendor_id"][chosen]
        vendor_name_col[bt_mask] = vendor_pool["vendor_name"][chosen]

    # Step 17: Transaction IDs (UUID4)
    transaction_ids = [str(uuid.uuid4()) for _ in range(n)]

    # Build DataFrame
    df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "user_id": user_id,
        "user_name": user_name,
        "user_email": user_email,
        "user_country": user_country,
        "user_membership_tier": user_membership_tier,
        "booking_type": booking_type,
        "booking_date": booking_date,
        "booking_timestamp": booking_timestamp,
        "destination_country": dest_country,
        "destination_city": dest_city,
        "origin_city": origin_city,
        "booking_status": booking_status,
        "payment_method": payment_method,
        "currency": currency,
        "base_amount": base_amount,
        "tax_amount": tax_amount,
        "discount_amount": discount_amount,
        "total_amount": total_amount,
        "promo_code": promo_code,
        "platform": platform,
        "device_os": device_os,
        "app_version": app_version,
        "session_duration_seconds": session_duration,
        "is_repeat_booking": is_repeat,
        "rating": rating,
        "review_text": review_text,
        "cancellation_reason": cancellation_reason,
        "vendor_id": vendor_id_col,
        "vendor_name": vendor_name_col,
    })

    return df


# ---------------------------------------------------------------------------
# Data quality validation
# ---------------------------------------------------------------------------

def validate_chunk(df: pd.DataFrame, chunk_idx: int) -> list:
    """Validate data quality rules. Returns list of failure dicts."""
    failures = []

    def check_rule(name: str, condition_series):
        n_fail = (~condition_series).sum()
        if n_fail > 0:
            failures.append({
                "chunk": chunk_idx,
                "rule": name,
                "failed_rows": int(n_fail),
            })

    # Rule 1: total_amount == round(base + tax - discount, 2)
    expected_total = np.round(df["base_amount"] + df["tax_amount"] - df["discount_amount"], 2)
    expected_total = np.maximum(expected_total, 0)
    check_rule(
        "total_amount_equals_base_plus_tax_minus_discount",
        np.isclose(df["total_amount"], expected_total, atol=0.01),
    )

    # Rule 2: cancellation_reason NULL when status not cancelled/refunded
    non_cancel = ~df["booking_status"].isin(["cancelled", "refunded"])
    check_rule(
        "cancellation_reason_null_for_non_cancelled",
        df.loc[non_cancel, "cancellation_reason"].isna(),
    )

    # Rule 3: promo_code not null implies discount > 0
    has_promo = df["promo_code"].notna()
    if has_promo.any():
        check_rule(
            "promo_code_implies_positive_discount",
            df.loc[has_promo, "discount_amount"] > 0,
        )

    # Rule 4: origin_city is a valid city for user_country (vectorized)
    valid_origin = np.ones(len(df), dtype=bool)
    for country, cities in COUNTRY_CITY_MAP.items():
        mask = (df["user_country"] == country).values
        if mask.any():
            valid_origin[mask] = df.loc[mask, "origin_city"].isin(cities).values
    check_rule("origin_city_valid_for_user_country", pd.Series(valid_origin, index=df.index))

    # Rule 5: destination_city is a valid city for destination_country (vectorized)
    valid_dest = np.ones(len(df), dtype=bool)
    for country, cities in COUNTRY_CITY_MAP.items():
        mask = (df["destination_country"] == country).values
        if mask.any():
            valid_dest[mask] = df.loc[mask, "destination_city"].isin(cities).values
    check_rule("destination_city_valid_for_destination_country", pd.Series(valid_dest, index=df.index))

    # Rule 6: No duplicate transaction_id (within chunk)
    check_rule(
        "no_duplicate_transaction_id",
        ~df["transaction_id"].duplicated(),
    )

    # Rule 7: booking_timestamp date matches booking_date
    ts_dates = pd.to_datetime(df["booking_timestamp"]).dt.date
    # Note: timestamps are UTC-converted, so date may shift by 1 day from local date.
    # We accept +/- 1 day due to timezone conversion
    booking_dates = pd.to_datetime(df["booking_date"])
    ts_date_series = pd.to_datetime(ts_dates)
    date_diff = (ts_date_series - booking_dates).dt.days.abs()
    check_rule("booking_timestamp_date_within_1day", date_diff <= 1)

    # Rule 8: rating NULL when status is pending or in_progress
    pending_mask = df["booking_status"].isin(["pending", "in_progress"])
    if pending_mask.any():
        check_rule(
            "rating_null_for_pending_in_progress",
            df.loc[pending_mask, "rating"].isna(),
        )

    return failures


# ---------------------------------------------------------------------------
# Chunk sizing calibration
# ---------------------------------------------------------------------------

def calibrate_chunk_size(
    rng: np.random.Generator,
    user_pool: dict,
    vendor_pool: dict,
    output_dir: str,
    target_mb: float = 90.0,
    pilot_rows: int = 50_000,
) -> int:
    """Generate a pilot chunk, measure size, extrapolate rows per chunk."""
    print(f"\nCalibrating chunk size (pilot: {pilot_rows:,} rows, target: {target_mb}MB)...")

    pilot_df = generate_chunk(pilot_rows, rng, user_pool, vendor_pool)
    pilot_path = os.path.join(output_dir, "_pilot.parquet")

    table = pa.Table.from_pandas(pilot_df, schema=PARQUET_SCHEMA, preserve_index=False)
    pq.write_table(table, pilot_path, compression="snappy")

    pilot_size_bytes = os.path.getsize(pilot_path)
    pilot_size_mb = pilot_size_bytes / (1024 * 1024)

    rows_per_mb = pilot_rows / pilot_size_mb
    rows_per_chunk = int(rows_per_mb * target_mb)

    # Enforce bounds
    rows_per_chunk = max(100_000, min(rows_per_chunk, 2_000_000))

    # Cleanup pilot
    os.remove(pilot_path)
    del pilot_df
    gc.collect()

    estimated_chunk_mb = rows_per_chunk / rows_per_mb
    print(f"  Pilot size: {pilot_size_mb:.2f}MB for {pilot_rows:,} rows")
    print(f"  Rows per MB: {rows_per_mb:,.0f}")
    print(f"  Calibrated chunk size: {rows_per_chunk:,} rows (~{estimated_chunk_mb:.1f}MB)")

    return rows_per_chunk


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------

def write_parquet(df: pd.DataFrame, chunk_idx: int, output_dir: str) -> tuple:
    """Write a DataFrame as a Snappy-compressed Parquet file. Returns (path, size_bytes)."""
    filename = f"voyagehub_transactions_part_{chunk_idx:03d}.parquet"
    filepath = os.path.join(output_dir, filename)

    table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
    pq.write_table(table, filepath, compression="snappy")

    size_bytes = os.path.getsize(filepath)
    return filepath, size_bytes


# ---------------------------------------------------------------------------
# Email hashing (privacy mode)
# ---------------------------------------------------------------------------

def hash_emails(df: pd.DataFrame) -> pd.DataFrame:
    """Replace user_email with SHA-256 hash for privacy/demo mode."""
    df["user_email"] = df["user_email"].apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest() if isinstance(x, str) else x
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate VoyageHub synthetic travel dataset"
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "prod"],
        default="smoke",
        help="smoke=1.5M rows (quick test), prod=60M rows (full dataset)",
    )
    parser.add_argument(
        "--hash-emails",
        action="store_true",
        help="Hash user emails with SHA-256 for privacy/demo mode",
    )
    parser.add_argument(
        "--output-dir",
        default="voyagehub_dataset",
        help="Output directory for Parquet files (default: voyagehub_dataset)",
    )
    args = parser.parse_args()

    # Configuration
    target_rows = 1_500_000 if args.mode == "smoke" else 60_000_000
    run_id = str(uuid.uuid4())[:8]
    start_time = datetime.now(timezone.utc)

    print("=" * 70)
    print("  VOYAGEHUB SYNTHETIC DATASET GENERATOR")
    print("=" * 70)
    print(f"  Run ID       : {run_id}")
    print(f"  Mode         : {args.mode}")
    print(f"  Target rows  : {target_rows:,}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Hash emails  : {args.hash_emails}")
    print(f"  Started at   : {start_time.isoformat()}")
    print("=" * 70)

    # Setup
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(42)

    # Generate pools
    user_pool = generate_user_pool(rng, n_users=2_000_000)
    vendor_pool = generate_vendor_pool(rng, n_vendors=5_000)

    # Calibrate chunk size
    rows_per_chunk = calibrate_chunk_size(rng, user_pool, vendor_pool, output_dir)

    total_chunks = (target_rows + rows_per_chunk - 1) // rows_per_chunk
    print(f"\nWill generate {total_chunks} chunks of ~{rows_per_chunk:,} rows each")
    print(f"Estimated total: {total_chunks * rows_per_chunk:,} rows")
    print("-" * 70)

    # Generation loop
    total_generated = 0
    total_size_bytes = 0
    chunk_index = 1
    all_audit_results = []

    gen_start = time.time()

    while total_generated < target_rows:
        chunk_start = time.time()
        this_chunk_size = min(rows_per_chunk, target_rows - total_generated)

        # Generate
        df = generate_chunk(this_chunk_size, rng, user_pool, vendor_pool)

        # Validate
        chunk_failures = validate_chunk(df, chunk_index)
        all_audit_results.extend(chunk_failures)

        # Hash emails if requested
        if args.hash_emails:
            df = hash_emails(df)

        # Write
        filepath, size_bytes = write_parquet(df, chunk_index, output_dir)
        size_mb = size_bytes / (1024 * 1024)
        total_size_bytes += size_bytes
        total_generated += len(df)

        chunk_elapsed = time.time() - chunk_start
        rows_per_sec = len(df) / chunk_elapsed if chunk_elapsed > 0 else 0

        fail_str = f" [{len(chunk_failures)} rule failures]" if chunk_failures else ""
        print(
            f"  Chunk {chunk_index:03d}/{total_chunks:03d} written: "
            f"{len(df):>10,} rows | {size_mb:6.1f}MB | "
            f"{rows_per_sec:,.0f} rows/sec | {chunk_elapsed:.1f}s{fail_str}"
        )

        if size_mb > 95:
            print(f"  WARNING: Chunk exceeds 95MB limit ({size_mb:.1f}MB)")

        del df
        if chunk_index % 10 == 0:
            gc.collect()

        chunk_index += 1

    gen_elapsed = time.time() - gen_start
    end_time = datetime.now(timezone.utc)
    total_size_mb = total_size_bytes / (1024 * 1024)
    total_size_gb = total_size_bytes / (1024 ** 3)
    avg_chunk_mb = total_size_mb / (chunk_index - 1) if chunk_index > 1 else 0
    overall_rows_per_sec = total_generated / gen_elapsed if gen_elapsed > 0 else 0
    n_rules_failed = len(all_audit_results)
    n_rules_passed = 8 * (chunk_index - 1) - n_rules_failed

    # Write audit report
    audit_path = os.path.join(output_dir, "audit_report.json")
    with open(audit_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "total_chunks_validated": chunk_index - 1,
            "total_rules_checked": 8 * (chunk_index - 1),
            "total_failures": n_rules_failed,
            "failures": all_audit_results,
        }, f, indent=2)

    # Write run summary
    summary = {
        "run_id": run_id,
        "mode": args.mode,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "runtime_minutes": round(gen_elapsed / 60, 2),
        "target_rows": target_rows,
        "total_rows_generated": total_generated,
        "total_files_written": chunk_index - 1,
        "total_size_bytes": total_size_bytes,
        "total_size_gb": round(total_size_gb, 3),
        "average_chunk_size_mb": round(avg_chunk_mb, 1),
        "rows_per_second": round(overall_rows_per_sec),
        "rows_per_chunk": rows_per_chunk,
        "hash_emails": args.hash_emails,
        "quality_rules_passed": n_rules_passed,
        "quality_rules_failed": n_rules_failed,
        "estimated_storage_footprint_gb": round(total_size_gb, 3),
        "max_runtime_target_minutes": 30 if args.mode == "prod" else 5,
    }

    summary_path = os.path.join(output_dir, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Final summary
    print()
    print("=" * 70)
    print("  GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Run ID              : {run_id}")
    print(f"  Mode                : {args.mode}")
    print(f"  Total rows          : {total_generated:,}")
    print(f"  Total files         : {chunk_index - 1}")
    print(f"  Total size          : {total_size_mb:,.1f}MB ({total_size_gb:.3f}GB)")
    print(f"  Avg chunk size      : {avg_chunk_mb:.1f}MB")
    print(f"  Runtime             : {gen_elapsed:.1f}s ({gen_elapsed/60:.1f} min)")
    print(f"  Throughput          : {overall_rows_per_sec:,.0f} rows/sec")
    print(f"  Quality rules       : {n_rules_passed} passed, {n_rules_failed} failed")
    print(f"  Storage footprint   : {total_size_gb:.3f}GB")
    print(f"  Audit report        : {audit_path}")
    print(f"  Run summary         : {summary_path}")
    print("=" * 70)

    if n_rules_failed > 0:
        print(f"\nWARNING: {n_rules_failed} data quality rule(s) failed. Check audit_report.json.")
        for f_item in all_audit_results:
            print(f"  - Chunk {f_item['chunk']:03d}: {f_item['rule']} ({f_item['failed_rows']} rows)")


if __name__ == "__main__":
    main()
