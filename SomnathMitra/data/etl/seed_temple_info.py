"""
Seeds clean.temple_info from manually-curated content sourced from somnath.org.

Unlike sync_pois.py / sync_restaurants.py, this isn't transforming a raw MongoDB
collection — the "raw" source here is the official website itself, fetched and
paraphrased by hand into the documents below. Re-running is idempotent (upsert
on `key`), so updating a fact means editing the DOCS list and re-running.

Run with: python etl/seed_temple_info.py
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_clean_db  # noqa: E402

TODAY = date.today().isoformat()

DOCS = [
    {
        "key": "history_significance",
        "category": "history",
        "title": "History & Significance of Somnath Temple",
        "source_url": "https://somnath.org/jay-somnath",
        "verified": True,
        "content": (
            "Somnath is regarded as the first among the twelve Jyotirlingas of Shiva and is "
            "referenced in ancient texts including the Skanda Purana and Shrimad Bhagavatam. "
            "Tradition holds the Moon God (Chandra) was freed from a curse here by Shiva. "
            "The temple is said to have been rebuilt across several eras — in gold, silver, wood, "
            "and stone — with the stone-era rebuilding credited to King Bhimadeva. It was destroyed "
            "and rebuilt multiple times through history; the current (7th) structure was initiated "
            "under Sardar Vallabhbhai Patel after independence, with a ~155-foot shikhara (tower). "
            "The temple sits on the Arabian Sea coast in Prabhas Patan, Gujarat, and the site is "
            "also associated with the end of Krishna's earthly life."
        ),
    },
    {
        "key": "darshan_timings",
        "category": "visit_info",
        "title": "Darshan Timings, Aarti & Evening Show",
        "source_url": "https://somnath.org/somnath-darshan/, https://somnath.org/faq/",
        "verified": True,
        "content": (
            "Temple is open for darshan 6:00 AM to 10:00 PM daily. Aarti takes place three times "
            "daily: 7:00 AM, 12:00 noon, and 7:00 PM. Entry is free. A sound-and-light show "
            "('Jay Somnath') runs nightly, though the two official pages disagree on the exact "
            "time (one says 7:45-8:45 PM, the FAQ page says 8-9 PM) — treat as approximate until "
            "confirmed. The temple is lit up every evening."
        ),
        "caveats": ["Light-show time conflicts between /somnath-darshan/ (7:45-8:45pm) and /faq/ (8-9pm) — needs verification."],
    },
    {
        "key": "visitor_rules",
        "category": "visit_info",
        "title": "Visitor Rules, Dress Code & Prohibited Items",
        "source_url": "https://somnath.org/faq/",
        "verified": True,
        "content": (
            "Footwear must be removed before entry; free shoe storage is provided. All electronic "
            "devices/gadgets (including phones and cameras) are prohibited inside and must be left "
            "in free lockers — photography is not allowed, but the trust sells official high-resolution "
            "photos. Dress should be respectful; mini-skirts and similarly casual/disrespectful attire "
            "are discouraged. Smoking is strictly forbidden on the premises. Wheelchairs, golf carts, "
            "and lift access are provided free of charge for elderly/disabled visitors."
        ),
    },
    {
        "key": "pilgrim_facilities",
        "category": "visit_info",
        "title": "Guest Houses, Dormitories & On-site Facilities",
        "source_url": "https://somnath.org/pilgrim-facilities, https://somnath.org/somnath-darshan/",
        "verified": True,
        "content": (
            "The trust runs its own accommodation (200+ rooms across a VIP guest house and 18 other "
            "guest houses), a Tirth Darshan bus service to nearby temples (departs 8:30 AM and 3:30 PM, "
            "nominal fee), a restaurant open until 11 PM, cradle facilities, and free wheelchairs/lockers. "
            "Payment by cash, card, or UPI is accepted."
        ),
        "items": [
            {"name": "Sagar Darshan Atithi Gruh", "note": "sea-view; 4-bed suite ~Rs 5,000, 2-bed suite ~Rs 4,600, premier/super-deluxe options", "phone": "02876-233533"},
            {"name": "Lilavati Atithi Gruh", "note": "non-AC ~Rs 750, AC ~Rs 950, AC suite ~Rs 1,568, 4-bed AC ~Rs 1,904", "phone": "02876-233033"},
            {"name": "Maheshwari Samaj Atithi Gruh", "note": "non-AC ~Rs 750, AC ~Rs 950, AC suite ~Rs 1,568", "phone": "02876-233130"},
            {"name": "Non-AC Dormitory", "note": "~Rs 90 per person", "phone": "02876-233433"},
            {"name": "AC Dormitory", "note": "~Rs 200 per person", "phone": "+91-6357571008"},
        ],
    },
    {
        "key": "nearest_places",
        "category": "nearby",
        "title": "Nearby Places & Attractions",
        "source_url": "https://somnath.org/nearest-places/, https://somnath.org/faq/",
        "verified": True,
        "content": "Places of interest within a short distance of the temple, per the official site.",
        "items": [
            {"name": "Bhalka Tirth", "distance_km": 5, "note": "site where Krishna was struck by the hunter Jara's arrow"},
            {"name": "Dehotsarg Tirth", "distance_km": 1.5, "note": "where Krishna is said to have departed his earthly form"},
            {"name": "Geeta Mandir", "distance_km": None, "note": "18 pillars inscribed with Bhagavad Gita chapters"},
            {"name": "Mahaprabhuji Baithak", "distance_km": None, "note": "one of 84 meditation seats of Mahaprabhuji"},
            {"name": "Shri Ram Mandir", "distance_km": None, "note": "built by Shree Somnath Trust in 2017, marble idols"},
            {"name": "Ahilyabai Temple", "distance_km": None, "note": "built 1783 by Queen Ahilyabai Holkar"},
            {"name": "Rudreshwar", "distance_km": None, "note": "13th-century Nagar-style temple"},
            {"name": "Avadhuteshwar Temple", "distance_km": None, "note": "shivling said to be established by King Vraj"},
            {"name": "Veneshwar", "distance_km": None, "note": "13th-14th century temple with a stepwell"},
            {"name": "Prachi Tirth", "distance_km": None, "note": "site for ancestral rituals"},
            {"name": "Gauri Kund", "distance_km": None, "note": "sacred water pool dedicated to Gauri"},
            {"name": "Triveni Ghat", "distance_km": None, "note": "confluence of the Hiran, Kapila and (mythical) Saraswati rivers"},
            {"name": "Bhadrakali Rock Inscription", "distance_km": None, "note": "1169 AD inscription referencing the temple"},
            {"name": "Daityasudan Temple", "distance_km": None, "note": "black marble deity idol"},
            {"name": "Shav No Timbo", "distance_km": None, "note": "post-Harappan era archaeological mound"},
            {"name": "Somnath Museum", "distance_km": None, "note": "relics from 10th-12th century temple structures"},
            {"name": "Matri Vav", "distance_km": None, "note": "ancient carved stepwell"},
            {"name": "Sun Temple (Triveni)", "distance_km": None, "note": "ruins of a solar-worship structure"},
            {"name": "Old Jain Temple", "distance_km": None, "note": "carved interior ceiling"},
            {"name": "Jain Temples (group of 9)", "distance_km": None, "note": "principal shrine to Chandraprabhu Swami"},
            {"name": "Old Caves", "distance_km": None, "note": "structures dated 1st-2nd century AD"},
            {"name": "Haveli Sheri", "distance_km": None, "note": "street of traditional wooden architecture"},
            {"name": "Veraval Gate", "distance_km": None, "note": "11th-12th century west-facing gate"},
            {"name": "Diu (airport + beaches)", "distance_km": 85, "note": "nearest airport"},
            {"name": "Asiatic Lion Sanctuary (Gir)", "distance_km": 65, "note": "wildlife sanctuary"},
            {"name": "Dwarka Temple", "distance_km": 230, "note": "another major Krishna pilgrimage site"},
            {"name": "Somnath Railway Station", "distance_km": 0.5, "note": "nearest railway station"},
        ],
    },
    {
        "key": "festivals_calendar",
        "category": "events",
        "title": "Festivals & Events",
        "source_url": "https://somnath.org/festivals",
        "verified": False,
        "content": (
            "The official festivals page lists specific date ranges that look tied to a particular "
            "year's calendar (e.g. a several-day range for Mahashivratri) rather than fixed annual "
            "dates. Store the list below as 'which festivals the trust marks', but re-fetch this page "
            "each year rather than trusting these exact dates long-term."
        ),
        "items": [
            {"name": "Somnath Swabhiman Parv", "when": "January 11"},
            {"name": "Makar Sankranti", "when": "January 14"},
            {"name": "Republic Day", "when": "January 26"},
            {"name": "Mahashivratri", "when": "February 15-26 (per site's current-year listing)"},
            {"name": "Vaidik Holika Dahan", "when": "March 2-24 (per site's current-year listing)"},
            {"name": "Golokdham Utsav", "when": "March 19-30 (per site's current-year listing)"},
            {"name": "Ram Janmotsav", "when": "March/April"},
            {"name": "Somnath Sthapnadin (temple founding day)", "when": "May 11"},
            {"name": "Parshuramji Janmotsav", "when": "April/May"},
            {"name": "Ganga Dussehra", "when": "May 25"},
            {"name": "International Yoga Day", "when": "June 21"},
            {"name": "Shravan Prarambh (start of Shravan month)", "when": "July 29"},
            {"name": "Krishna Janmastami", "when": "August 26"},
            {"name": "Vijayadashami", "when": "October 12"},
            {"name": "Kartik Purnima", "when": "November 15"},
            {"name": "Geeta Jayanti", "when": "December 11"},
        ],
    },
    {
        "key": "faqs",
        "category": "faq",
        "title": "Frequently Asked Questions",
        "source_url": "https://somnath.org/faq/",
        "verified": True,
        "content": "Official FAQ list, paraphrased.",
        "items": [
            {"q": "Who can pray at the temple?", "a": "Anyone with faith in Hinduism may offer prayers."},
            {"q": "What are the darshan hours?", "a": "6:00 AM to 10:00 PM; aarti at 7 AM, noon, and 7 PM; evening light show around 8-9 PM."},
            {"q": "Are facilities available for elderly/disabled visitors?", "a": "Wheelchairs, golf carts, and lift access are free."},
            {"q": "Is transport to nearby temples offered?", "a": "A Tirth Darshan bus runs at 8:30 AM and 3:30 PM for a nominal fee."},
            {"q": "Is entry free?", "a": "Yes, entry is completely free."},
            {"q": "Can I watch darshan online?", "a": "Yes, via the Live Darshan page on the website."},
            {"q": "What are the major festivals?", "a": "Shravan month, Mahashivratri, Golokdham Utsav, Kartik Purnima Fair, Somnath Sthapana Divas, among others."},
            {"q": "Is havan (fire ritual) available?", "a": "Yes, bookable through the General Manager's office."},
            {"q": "Must I remove footwear?", "a": "Yes; free shoe storage is provided."},
            {"q": "Can I bring electronics?", "a": "No, gadgets are prohibited inside but can be stored free in lockers."},
            {"q": "Is there a dress code?", "a": "Respectful attire is expected; mini-skirts etc. discouraged; smoking is forbidden."},
            {"q": "Is photography allowed?", "a": "No; official high-resolution photos can be purchased instead."},
            {"q": "How can I donate?", "a": "By cheque, cash, or online payment; receipts available from staff or donation boxes."},
            {"q": "Can rooms be booked online?", "a": "Yes, through the accommodation section of the website."},
            {"q": "Are baby/cradle facilities available?", "a": "Yes, at all guest houses."},
            {"q": "Do guest-house restaurants exist?", "a": "Yes, serving Gujarati, Punjabi, Chinese, and general Indian cuisine."},
            {"q": "Are banquet halls available?", "a": "Yes; contact 02876-233533 or +91-9428214914."},
            {"q": "What's the nearest airport/station?", "a": "Diu airport (~85 km); Somnath railway station (~0.5 km)."},
            {"q": "What tourist attractions are nearby?", "a": "Several temples within 20 km, Gir Lion Sanctuary (~65 km), Diu beaches (~85 km), Dwarka (~230 km)."},
        ],
    },
    {
        "key": "social_activities",
        "category": "about_trust",
        "title": "Shree Somnath Trust — Social & Charitable Activities",
        "source_url": "https://somnath.org/social-activities/",
        "verified": True,
        "content": (
            "Beyond temple operations, the trust runs charitable programs: tree-plantation drives, two "
            "gaushalas (cow shelters) with subsidized veterinary services, twice-daily food donation "
            "(Prasad food program) and child-nutrition efforts, free monthly medical/dental/eye camps, "
            "a physiotherapy center, skill-development scholarships, disaster relief (COVID-19, Cyclone "
            "Tauktae, floods), stepwell/water-reservoir renovation, and waste-management work that earned "
            "it recognition as a 'Cleanest Iconic Place' and plastic-free zone."
        ),
    },
    {
        "key": "contact_info",
        "category": "contact",
        "title": "Contact Details",
        "source_url": "https://somnath.org/contact/",
        "verified": True,
        "content": "Prabhas Patan, Dist. Gir Somnath - 362268, Gujarat, India.",
        "items": [
            {"dept": "General inquiry", "phone": "+91-94282 14914"},
            {"dept": "Main office", "phone": "+91-2876-232694"},
            {"dept": "Pooja Vidhi & donations counter", "phone": "+91-94282 14823"},
            {"dept": "Temple officer", "phone": "+91-94262 87659"},
            {"dept": "Guest house central booking (8am-9pm)", "phone": "+91-2876-231212"},
            {"dept": "Technical support", "phone": "+91-9428214993"},
            {"dept": "Ahmedabad office", "phone": "+91-79-22686335 / +91-79-22686442"},
            {"dept": "Email (Pooja Vidhi)", "email": "sompp@somnath.org"},
            {"dept": "Email (IT)", "email": "itsuper@somnath.org"},
        ],
    },
    {
        "key": "heritage_and_temple_walks",
        "category": "visit_info",
        "title": "Heritage Walk & Temple Walk",
        "source_url": "https://somnath.org/heritage-walk, https://somnath.org/temple-walk",
        "verified": False,
        "content": (
            "Both official pages only publish leaflet images with no extractable text — route, stops, "
            "timing, and purpose are unknown from the scrape. Needs manual entry (read the leaflet "
            "images directly, or contact the trust) before this can answer any real question."
        ),
    },
]


def run():
    clean_db = get_clean_db()
    for doc in DOCS:
        doc = {**doc, "last_updated": TODAY, "schema_version": 1}
        clean_db["temple_info"].replace_one({"key": doc["key"]}, doc, upsert=True)
    print(f"seeded {len(DOCS)} temple_info documents")


if __name__ == "__main__":
    run()
