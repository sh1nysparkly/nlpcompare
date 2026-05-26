#!/usr/bin/env python3
"""
NLP category experiments for /vacation-packages/tours.
Tests variations of section copy to measure Sightseeing Tours category impact.
"""

import json
import time
import requests

BRIDGE_URL = "https://ghzfrxxevjjfgpxvmahy.supabase.co/functions/v1/bridge"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdoemZyeHhldmpqZmdweHZtYWh5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgxNzk4OTgsImV4cCI6MjA5Mzc1NTg5OH0.VPJTvdCvU217QmQBjm3ym8ZOoCgyBY-VpLpPdhefa04"

WATCH_CATEGORIES = [
    "Guided Tours & Escorted Tours",
    "Sightseeing Tours",
    "Travel Agencies & Services",
    "Adventure Travel",
    "Tourist Destinations",
    "Vacation Offers",
    "Luxury Travel",
]


def call_nlp_classify(text):
    """Call bridge NLP classify and return categories dict."""
    resp = requests.post(
        BRIDGE_URL,
        headers={
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Content-Type": "application/json",
            "apikey": SUPABASE_ANON_KEY,
        },
        json={"tool": "nlp_classify", "params": {"text": text}},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data


def extract_categories(result):
    """Pull category name -> confidence from NLP result."""
    cats = {}
    raw_cats = result.get("categories", [])
    for c in raw_cats:
        name = c.get("name", "")
        # strip leading path, keep leaf
        leaf = name.split("/")[-1].strip()
        full = name.strip().lstrip("/")
        conf = c.get("confidence", 0)
        cats[full] = conf
    return cats


def score_text(label, text):
    """Score text and print watched categories."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {label}")
    print(f"  Word count: {len(text.split())}")
    print(f"{'='*70}")

    result = call_nlp_classify(text)
    cats = extract_categories(result)

    print(f"\n  {'Category':<50} {'Confidence':>10}")
    print(f"  {'-'*60}")

    for watch in WATCH_CATEGORIES:
        found = False
        for full_path, conf in cats.items():
            if watch.lower() in full_path.lower():
                pct = conf * 100
                marker = " ***" if watch == "Sightseeing Tours" else ""
                marker = " ++++" if watch == "Guided Tours & Escorted Tours" else marker
                print(f"  {full_path:<50} {pct:>8.1f}%{marker}")
                found = True
        if not found:
            print(f"  {watch:<50} {'--':>10}")

    print(f"\n  All categories returned:")
    for path, conf in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {path:<55} {conf*100:>6.1f}%")

    return cats


# ============================================================================
# Page text assembly
# ============================================================================

# Sections that DON'T change across experiments (static scaffolding)

HERO = "Find Guided Tours Wherever You Want to Go"

TOP_PARTNERS = "Top Tour Partners. Multi-day guided and escorted tours."

# FAQ section (unchanged)
FAQ = """Guided Tour FAQs.
What is the definition of a tour? A tour is a structured trip involving a planned itinerary, typically organized and led by a professional guide or tour company.
What is the difference between escorted and guided tours? Escorted tours provide a dedicated tour director who travels with the group throughout the entire trip, while guided tours may use local guides at specific stops or attractions.
What is the best tour company in Canada? Some of the top tour companies operating in Canada include Trafalgar, Insight Vacations, Collette, and G Adventures. AMA Travel partners with leading tour operators to bring Albertans a curated selection of guided and escorted tour options.
Is it cheaper to travel with a tour company? In many cases, yes. Tour companies negotiate group rates on hotels, transport, and attractions, passing savings on to travellers. Packages often include meals, transfers, and tips, making it easier to budget.
What is the best site for tour packages? AMA Travel offers a wide selection of multi-day tour packages from trusted tour operators. Our Alberta-based travel agents help you compare options and find the right fit.
What is a multi-day tour? A multi-day tour is an organized trip lasting two or more days that follows a set itinerary, with accommodation, transportation, and guided activities included."""

CURATED = "Curated Guided Tour Packages. Browse our curated collections of guided vacations, grouped by style and interest."

# Tour product cards (static - these are deal cards with supplier/pricing info)
TOUR_CARDS = """Trafalgar European Discovery 14 days from $3,299 per person.
Insight Vacations Country Roads of Italy 10 days from $4,195 per person.
Collette Exploring South Africa 14 days from $5,099 per person.
G Adventures Costa Rica Quest 9 days from $1,799 per person.
Globus Hawaiian Adventure 10 days from $3,599 per person.
Contiki European Discovery 14 days from $2,499 per person."""

# Footer cards (static)
FOOTER = "Vacation Packages. Flights. Hotels. Car Rentals."


def build_page(explore_section, why_book_section):
    """Assemble full page text from static + variable sections."""
    parts = [
        HERO,
        TOP_PARTNERS,
        explore_section,
        why_book_section,
        CURATED,
        TOUR_CARDS,
        FAQ,
        FOOTER,
    ]
    return "\n\n".join(parts)


# ============================================================================
# Experiment variations
# ============================================================================

# --- EXPLORE SECTION VARIATIONS ---

EXPLORE_V4_BASELINE = """Explore Multi-Day Journeys by Style.
Whether you prefer a relaxed escorted bus tour, a luxury guided vacation, or a small-group adventure, AMA Travel has something for everyone.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

EXPLORE_A = """Multi-Day Tour Styles.
Whether you prefer a relaxed escorted bus tour, a luxury guided vacation, or a small-group adventure, AMA Travel has an itinerary for every travel style.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

EXPLORE_B = """Choose Your Multi-Day Tour Style.
From escorted bus tours to luxury guided vacations, find the itinerary that fits your travel style.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

EXPLORE_C = """Multi-Day Tours by Travel Style.
From escorted coach tours to luxury guided vacations, find an itinerary that fits how you like to travel.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents build a custom itinerary around your interests, booking each detail individually."""

EXPLORE_D = """Find Multi-Day Tours by Style.
Escorted coach tours, guided vacations, small-group itineraries -- whatever your travel style, we have a tour to match.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents build a custom itinerary around your interests, booking each detail individually."""

EXPLORE_E = """Multi-Day Guided Tour Styles.
Escorted coach tours, guided group vacations, and small-group itineraries tailored to how you like to travel.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents build a custom itinerary around your interests, booking each detail individually."""


# --- WHY BOOK SECTION VARIATIONS ---

WHY_BOOK_V4_BASELINE = """Plan Your Perfect Tour with Travel Experts.
Why Book Multi-Day Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading travel tour companies to bring Albertans offers that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on tour packages, savings on travel medical insurance, and insider tips from our destination experts. Whether you're looking for week-long Europe tour packages or luxury escorted vacations across Asia or Africa, you can count on AMA's expertise to help you book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and guided multi day tours tailored to your style and interests.
Expert Advice. Our Alberta-based travel agents work with top tour providers to match you with the right trip every time."""

WHY_BOOK_A = """Plan Your Perfect Tour with Travel Experts.
Why Book Multi-Day Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour operators to bring Albertans guided and escorted tour itineraries that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on tour packages, savings on travel medical insurance, and insider tips from our travel experts. Whether you're comparing week-long European tour itineraries or planning luxury escorted vacations across Asia or Africa, you can count on AMA's expertise to help you book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and guided multi-day tour itineraries tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour operators to match you with the right trip every time."""

WHY_BOOK_B = """Plan Your Perfect Tour with Travel Experts.
Why Book Guided Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour operators to build itineraries that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on guided tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European tour itineraries or planning escorted vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and guided multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour operators to match you with the right itinerary every time."""

WHY_BOOK_C = """Plan Your Perfect Tour with Travel Experts.
Why Book Guided Tours with AMA Travel.
Booking with AMA Travel means more than a multi-day tour package -- it means a trusted travel partner from start to finish. We work with leading tour operators to offer escorted and guided itineraries that are safe, reliable, and built for comfort.
AMA members enjoy exclusive perks: discounts on guided tour packages, savings on travel medical insurance, and direct access to our Alberta-based travel experts. Comparing European tour itineraries or planning an escorted vacation across Asia? Our agents match you with the right operator and the right trip. Match with a Travel Agent.
Endless Options. Escorted, hosted, and guided multi-day tours with detailed itineraries, tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour operators to match you with the right itinerary every time."""


# ============================================================================
# Run experiments
# ============================================================================

def main():
    results = {}

    experiments = [
        ("V4 Baseline", EXPLORE_V4_BASELINE, WHY_BOOK_V4_BASELINE),
        ("A: Drop 'Explore', add 'itinerary' to subtext", EXPLORE_A, WHY_BOOK_V4_BASELINE),
        ("B: 'Choose Your' + shorter subtext w/ itinerary", EXPLORE_B, WHY_BOOK_V4_BASELINE),
        ("C: 'Tours by Travel Style' + coach + itinerary", EXPLORE_C, WHY_BOOK_V4_BASELINE),
        ("D: 'Find' verb + itinerary in subtext", EXPLORE_D, WHY_BOOK_V4_BASELINE),
        ("E: 'Guided Tour Styles' heading", EXPLORE_E, WHY_BOOK_V4_BASELINE),
        ("F: Baseline explore + WhyBook A (itinerary swap)", EXPLORE_V4_BASELINE, WHY_BOOK_A),
        ("G: Baseline explore + WhyBook B (tighter, guided focus)", EXPLORE_V4_BASELINE, WHY_BOOK_B),
        ("H: Baseline explore + WhyBook C (strongest rewrite)", EXPLORE_V4_BASELINE, WHY_BOOK_C),
        ("I: Explore C + WhyBook B (best combo candidate)", EXPLORE_C, WHY_BOOK_B),
        ("J: Explore E + WhyBook C (max guided signal)", EXPLORE_E, WHY_BOOK_C),
        ("K: Explore D + WhyBook A (itinerary everywhere)", EXPLORE_D, WHY_BOOK_A),
    ]

    for label, explore, why_book in experiments:
        page_text = build_page(explore, why_book)
        cats = score_text(label, page_text)
        results[label] = cats
        time.sleep(1)

    # Summary table
    print("\n\n" + "="*90)
    print("  SUMMARY: Sightseeing Tours vs Guided Tours across all experiments")
    print("="*90)
    print(f"\n  {'Experiment':<55} {'Guided':>8} {'Sightsee':>8} {'Delta':>8}")
    print(f"  {'-'*80}")

    for label in [e[0] for e in experiments]:
        cats = results[label]
        guided = 0
        sightsee = 0
        for path, conf in cats.items():
            if "guided tours" in path.lower() or "escorted tours" in path.lower():
                guided = max(guided, conf)
            if "sightseeing" in path.lower():
                sightsee = max(sightsee, conf)
        delta = guided - sightsee
        print(f"  {label:<55} {guided*100:>7.1f}% {sightsee*100:>7.1f}% {delta*100:>+7.1f}")


if __name__ == "__main__":
    main()
