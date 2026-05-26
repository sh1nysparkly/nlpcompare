#!/usr/bin/env python3
"""
Round 2: Targeted NLP experiments for /vacation-packages/tours.
Testing specific levers identified in round 1:
- "guided" vs "multi-day" in heading positions
- removing sightseeing vocabulary (destination, explore, adventure, solo)
- card name variations
- copy tightness
"""

import json
import time
import requests

BRIDGE_URL = "https://ghzfrxxevjjfgpxvmahy.supabase.co/functions/v1/bridge"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdoemZyeHhldmpqZmdweHZtYWh5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgxNzk4OTgsImV4cCI6MjA5Mzc1NTg5OH0.VPJTvdCvU217QmQBjm3ym8ZOoCgyBY-VpLpPdhefa04"


def call_nlp_classify(text):
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
    return resp.json()


def extract_scores(result):
    guided = 0
    sightsee = 0
    all_cats = {}
    for c in result.get("categories", []):
        name = c["name"].strip().lstrip("/")
        conf = c["confidence"]
        all_cats[name] = conf
        if "guided tours" in name.lower() or "escorted" in name.lower():
            guided = max(guided, conf)
        if "sightseeing" in name.lower():
            sightsee = max(sightsee, conf)
    return guided, sightsee, all_cats


# ============================================================================
# Static scaffolding (same for all experiments)
# ============================================================================

HERO = "Find Guided Tours Wherever You Want to Go"
TOP_PARTNERS = "Top Tour Partners. Multi-day guided and escorted tours."

FAQ = """Guided Tour FAQs.
What is the definition of a tour? A tour is a structured trip involving a planned itinerary, typically organized and led by a professional guide or tour company.
What is the difference between escorted and guided tours? Escorted tours provide a dedicated tour director who travels with the group throughout the entire trip, while guided tours may use local guides at specific stops or attractions.
What is the best tour company in Canada? Some of the top tour companies operating in Canada include Trafalgar, Insight Vacations, Collette, and G Adventures. AMA Travel partners with leading tour operators to bring Albertans a curated selection of guided and escorted tour options.
Is it cheaper to travel with a tour company? In many cases, yes. Tour companies negotiate group rates on hotels, transport, and attractions, passing savings on to travellers. Packages often include meals, transfers, and tips, making it easier to budget.
What is the best site for tour packages? AMA Travel offers a wide selection of multi-day tour packages from trusted tour operators. Our Alberta-based travel agents help you compare options and find the right fit.
What is a multi-day tour? A multi-day tour is an organized trip lasting two or more days that follows a set itinerary, with accommodation, transportation, and guided activities included."""

CURATED = "Curated Guided Tour Packages. Browse our curated collections of guided vacations, grouped by style and interest."

TOUR_CARDS = """Trafalgar European Discovery 14 days from $3,299 per person.
Insight Vacations Country Roads of Italy 10 days from $4,195 per person.
Collette Exploring South Africa 14 days from $5,099 per person.
G Adventures Costa Rica Quest 9 days from $1,799 per person.
Globus Hawaiian Adventure 10 days from $3,599 per person.
Contiki European Discovery 14 days from $2,499 per person."""

FOOTER = "Vacation Packages. Flights. Hotels. Car Rentals."


def build_page(explore_section, why_book_section, curated=CURATED):
    parts = [HERO, TOP_PARTNERS, explore_section, why_book_section, curated, TOUR_CARDS, FAQ, FOOTER]
    return "\n\n".join(parts)


# ============================================================================
# Round 1 winner (G) as new baseline
# ============================================================================

EXPLORE_BASELINE = """Explore Multi-Day Journeys by Style.
Whether you prefer a relaxed escorted bus tour, a luxury guided vacation, or a small-group adventure, AMA Travel has something for everyone.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

WHY_BOOK_G = """Plan Your Perfect Tour with Travel Experts.
Why Book Guided Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour operators to build itineraries that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on guided tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European tour itineraries or planning escorted vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and guided multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour operators to match you with the right itinerary every time."""


# ============================================================================
# Targeted experiments
# ============================================================================

experiments = []

# R1 winner (G) as baseline for this round
experiments.append(("R2-Baseline (R1 winner G)", EXPLORE_BASELINE, WHY_BOOK_G))

# --- TEST 1: "Explore" verb removal (since it didn't move much in R1, but confirm) ---
EXPLORE_NO_EXPLORE = """Multi-Day Guided Tours by Style.
Whether you prefer a relaxed escorted bus tour, a luxury guided vacation, or a small-group adventure, AMA Travel has something for everyone.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

experiments.append(("T1: Remove 'Explore', add 'Guided' to H2", EXPLORE_NO_EXPLORE, WHY_BOOK_G))

# --- TEST 2: Card name swap -- remove "Solo" and "Adventure" (sightseeing terms) ---
EXPLORE_CARD_SWAP = """Explore Multi-Day Journeys by Style.
Whether you prefer a relaxed escorted bus tour, a luxury guided vacation, or a small-group adventure, AMA Travel has something for everyone.
Bus Tours.
Small Group Tours.
Escorted Tours.
Hosted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

experiments.append(("T2: Replace Solo/Adventure cards with Small Group/Hosted", EXPLORE_CARD_SWAP, WHY_BOOK_G))

# --- TEST 3: Both -- remove Explore + swap cards ---
EXPLORE_COMBO_1 = """Multi-Day Guided Tours by Style.
Whether you prefer a relaxed escorted bus tour, a luxury guided vacation, or a small-group adventure, AMA Travel has something for everyone.
Bus Tours.
Small Group Tours.
Escorted Tours.
Hosted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

experiments.append(("T3: Guided H2 + card swap (combo)", EXPLORE_COMBO_1, WHY_BOOK_G))

# --- TEST 4: "destination" removal from curated section ---
CURATED_NO_DEST = "Curated Guided Tour Packages. Browse our curated collections of guided vacations, organized by tour style and travel interest."

experiments.append(("T4: Remove 'grouped' from Curated subtext", EXPLORE_BASELINE, WHY_BOOK_G, CURATED_NO_DEST))

# --- TEST 5: Heading-only - "Guided Tour Styles" (compact for UX) ---
EXPLORE_COMPACT = """Guided Tour Styles.
Escorted bus tours, luxury guided vacations, and small-group adventures -- find the trip that fits how you travel.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

experiments.append(("T5: 'Guided Tour Styles' (compact UX heading)", EXPLORE_COMPACT, WHY_BOOK_G))

# --- TEST 6: "Tours by Style" -- minimal heading ---
EXPLORE_MINIMAL = """Tours by Style.
Escorted bus tours, luxury guided vacations, and small-group adventures -- find the trip that fits how you travel.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

experiments.append(("T6: 'Tours by Style' (minimal heading)", EXPLORE_MINIMAL, WHY_BOOK_G))

# --- TEST 7: Remove "journeys" from heading + add "guided" qualifier ---
EXPLORE_NO_JOURNEY = """Guided Tours by Travel Style.
Whether you prefer a relaxed escorted bus tour, a luxury guided vacation, or a small-group adventure, AMA Travel has tours for every travel style.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

experiments.append(("T7: 'Guided Tours by Travel Style'", EXPLORE_NO_JOURNEY, WHY_BOOK_G))

# --- TEST 8: Subtext rewrite -- remove "small-group adventure" (sightseeing trigger?) ---
EXPLORE_NO_ADVENTURE = """Explore Multi-Day Journeys by Style.
Whether you prefer a relaxed escorted bus tour, a luxury guided vacation, or a hosted group trip, AMA Travel has a tour for every travel style.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

experiments.append(("T8: Subtext: 'hosted group trip' replaces 'small-group adventure'", EXPLORE_NO_ADVENTURE, WHY_BOOK_G))

# --- TEST 9: Full combo -- Guided H2 + card swap + tighter subtext + no adventure ---
EXPLORE_FULL_COMBO = """Guided Tours by Travel Style.
Escorted bus tours, luxury guided vacations, and hosted group trips -- find the tour that fits how you travel.
Bus Tours.
Small Group Tours.
Escorted Tours.
Hosted Tours.
Custom Tour. Our expert agents build your perfect tour, booking each detail around your interests."""

experiments.append(("T9: Full combo (guided H2 + card swap + tight subtext)", EXPLORE_FULL_COMBO, WHY_BOOK_G))

# --- TEST 10: WhyBook -- try "Why Book with a Tour Operator" framing ---
WHY_BOOK_OPERATOR = """Plan Your Perfect Tour with Travel Experts.
Why Book Guided Tours with AMA Travel.
Booking with AMA Travel means a trusted partner from first call to final day. We work with leading tour operators to offer escorted and guided tours that are safe, reliable, and built for comfort.
AMA members enjoy exclusive perks: discounts on guided tour packages, savings on travel medical insurance, and direct access to our Alberta-based travel agents. Comparing European guided tours or planning an escorted coach tour through Asia? Our agents match you with the right operator and trip. Match with a Travel Agent.
Endless Options. Escorted, hosted, and guided multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour operators to find you the right guided tour every time."""

experiments.append(("T10: WhyBook -- 'tour operators' + 'coach tour' framing", EXPLORE_BASELINE, WHY_BOOK_OPERATOR))

# --- TEST 11: Full combo best explore + operator WhyBook ---
experiments.append(("T11: Full combo explore + operator WhyBook", EXPLORE_FULL_COMBO, WHY_BOOK_OPERATOR))

# --- TEST 12: Test "Browse" in curated section -- replace with "Compare" ---
CURATED_COMPARE = "Curated Guided Tour Packages. Compare our curated collections of guided vacations, organized by tour style and travel interest."

experiments.append(("T12: Curated 'Compare' instead of 'Browse'", EXPLORE_BASELINE, WHY_BOOK_G, CURATED_COMPARE))


def main():
    results = []

    for exp in experiments:
        if len(exp) == 3:
            label, explore, why_book = exp
            page_text = build_page(explore, why_book)
        else:
            label, explore, why_book, curated = exp
            page_text = build_page(explore, why_book, curated)

        result = call_nlp_classify(page_text)
        guided, sightsee, all_cats = extract_scores(result)
        results.append((label, guided, sightsee, all_cats))

        print(f"  {label:<60} G:{guided*100:>5.1f}%  S:{sightsee*100:>5.1f}%  gap:{(guided-sightsee)*100:>+5.1f}")
        time.sleep(0.8)

    # Summary
    print(f"\n{'='*95}")
    print(f"  ROUND 2 SUMMARY")
    print(f"{'='*95}")
    print(f"\n  {'Experiment':<60} {'Guided':>7} {'Sight':>7} {'Gap':>7} {'S vs BL':>7}")
    print(f"  {'-'*88}")

    baseline_s = results[0][2]
    for label, guided, sightsee, _ in results:
        s_delta = (sightsee - baseline_s) * 100
        print(f"  {label:<60} {guided*100:>6.1f}% {sightsee*100:>6.1f}% {(guided-sightsee)*100:>+6.1f} {s_delta:>+6.1f}")

    # Top 3 by lowest sightseeing
    print(f"\n  TOP 3 by lowest Sightseeing:")
    sorted_by_s = sorted(results, key=lambda x: x[2])
    for label, guided, sightsee, _ in sorted_by_s[:3]:
        print(f"    {label:<60} S:{sightsee*100:>5.1f}%  G:{guided*100:>5.1f}%")

    # Top 3 by best gap (guided - sightseeing)
    print(f"\n  TOP 3 by largest Guided-Sightseeing gap:")
    sorted_by_gap = sorted(results, key=lambda x: -(x[1]-x[2]))
    for label, guided, sightsee, _ in sorted_by_gap[:3]:
        print(f"    {label:<60} gap:{(guided-sightsee)*100:>+5.1f}  G:{guided*100:>5.1f}%  S:{sightsee*100:>5.1f}%")


if __name__ == "__main__":
    main()
