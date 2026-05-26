#!/usr/bin/env python3
"""
Round 3: Final combination experiments + edge-case tests.
Combining R1+R2 winners and testing a few more hypotheses.
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
# Static scaffolding
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

TOUR_CARDS = """Trafalgar European Discovery 14 days from $3,299 per person.
Insight Vacations Country Roads of Italy 10 days from $4,195 per person.
Collette Exploring South Africa 14 days from $5,099 per person.
G Adventures Costa Rica Quest 9 days from $1,799 per person.
Globus Hawaiian Adventure 10 days from $3,599 per person.
Contiki European Discovery 14 days from $2,499 per person."""

FOOTER = "Vacation Packages. Flights. Hotels. Car Rentals."


def build_page(explore, why_book, curated):
    parts = [HERO, TOP_PARTNERS, explore, why_book, curated, TOUR_CARDS, FAQ, FOOTER]
    return "\n\n".join(parts)


# ============================================================================
# Best pieces from R1 + R2
# ============================================================================

# R1 original baseline explore
EXPLORE_ORIG = """Explore Multi-Day Journeys by Style.
Whether you prefer a relaxed escorted bus tour, a luxury guided vacation, or a small-group adventure, AMA Travel has something for everyone.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

# R2 winner: T5 heading + subtext
EXPLORE_T5 = """Guided Tour Styles.
Escorted bus tours, luxury guided vacations, and small-group adventures -- find the trip that fits how you travel.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

# R1 winner: WhyBook B (became G baseline)
WHY_BOOK_BEST = """Plan Your Perfect Tour with Travel Experts.
Why Book Guided Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour operators to build itineraries that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on guided tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European tour itineraries or planning escorted vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and guided multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour operators to match you with the right itinerary every time."""

WHY_BOOK_ORIG = """Plan Your Perfect Tour with Travel Experts.
Why Book Multi-Day Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading travel tour companies to bring Albertans offers that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on tour packages, savings on travel medical insurance, and insider tips from our destination experts. Whether you're looking for week-long Europe tour packages or luxury escorted vacations across Asia or Africa, you can count on AMA's expertise to help you book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and guided multi day tours tailored to your style and interests.
Expert Advice. Our Alberta-based travel agents work with top tour providers to match you with the right trip every time."""

CURATED_ORIG = "Curated Guided Tour Packages. Browse our curated collections of guided vacations, grouped by style and interest."
CURATED_COMPARE = "Curated Guided Tour Packages. Compare our curated collections of guided vacations, organized by tour style and travel interest."

experiments = []

# --- BASELINES ---
experiments.append(("V4-orig (from optimization guide)", EXPLORE_ORIG, WHY_BOOK_ORIG, CURATED_ORIG))
experiments.append(("R2-best (T5 explore + G whybook)", EXPLORE_T5, WHY_BOOK_BEST, CURATED_ORIG))

# --- BEST COMBO: T5 heading + G whybook + Compare curated ---
experiments.append(("COMBO-1: T5+G+Compare", EXPLORE_T5, WHY_BOOK_BEST, CURATED_COMPARE))

# --- Test: what if we drop "multi-day" from H1 subtext too? ---
# Current: "Top Tour Partners. Multi-day guided and escorted tours."
# This is in the static scaffolding, so we'll build custom for this test
def build_page_custom(hero, top_partners, explore, why_book, curated):
    parts = [hero, top_partners, explore, why_book, curated, TOUR_CARDS, FAQ, FOOTER]
    return "\n\n".join(parts)

# --- Test: "Explore" replaced with action verbs that aren't sightseeing-coded ---
EXPLORE_FIND = """Find Your Guided Tour Style.
Escorted bus tours, luxury guided vacations, and small-group adventures -- match the trip to how you travel.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

experiments.append(("T1: 'Find Your Guided Tour Style'", EXPLORE_FIND, WHY_BOOK_BEST, CURATED_COMPARE))

# --- Test: "Pick" as verb ---
EXPLORE_PICK = """Pick Your Tour Style.
Escorted bus tours, luxury guided vacations, and small-group adventures -- find the guided tour that fits how you travel.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

experiments.append(("T2: 'Pick Your Tour Style'", EXPLORE_PICK, WHY_BOOK_BEST, CURATED_COMPARE))

# --- Test: WhyBook -- tighter, drop "itinerary" (since R1 showed it boosts sightseeing) ---
WHY_BOOK_NO_ITIN = """Plan Your Perfect Tour with Travel Experts.
Why Book Guided Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading guided tour providers to offer trips that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on guided tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European guided tours or planning escorted vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and guided multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top guided tour providers to match you with the right trip every time."""

experiments.append(("T3: WhyBook no 'itinerary' -- 'guided tour providers'", EXPLORE_T5, WHY_BOOK_NO_ITIN, CURATED_COMPARE))

# --- Test: What does "vacation" do? It could pull toward Vacation Offers ---
# Replace "guided vacations" with "guided tours" in the subtext
EXPLORE_NO_VACATION = """Guided Tour Styles.
Escorted bus tours, luxury guided group tours, and small-group adventures -- find the trip that fits how you travel.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

experiments.append(("T4: 'guided group tours' instead of 'guided vacations'", EXPLORE_NO_VACATION, WHY_BOOK_BEST, CURATED_COMPARE))

# --- Test: the word "trip" vs "tour" -- "trip" is less sightseeing-coded? ---
EXPLORE_TRIP = """Guided Tour Styles.
Escorted bus tours, luxury guided vacations, and small-group adventures -- find the guided trip that fits how you travel.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect guided trip with your interests solely in mind."""

experiments.append(("T5: 'guided trip' in subtext + end cap", EXPLORE_TRIP, WHY_BOOK_BEST, CURATED_COMPARE))

# --- Test: T5 heading with original body subtext (longer) ---
EXPLORE_T5_ORIG_SUB = """Guided Tour Styles.
Whether you prefer a relaxed escorted bus tour, a luxury guided vacation, or a small-group adventure, AMA Travel has something for everyone.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

experiments.append(("T6: T5 heading + original longer subtext", EXPLORE_T5_ORIG_SUB, WHY_BOOK_BEST, CURATED_COMPARE))

# --- ABLATION: just the heading change, nothing else ---
EXPLORE_HEADING_ONLY = """Guided Tour Styles.
Whether you prefer a relaxed escorted bus tour, a luxury guided vacation, or a small-group adventure, AMA Travel has something for everyone.
Bus Tours.
Solo Tours.
Adventure Tours.
Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind."""

experiments.append(("ABLATION: T5 heading only, orig WhyBook + curated", EXPLORE_HEADING_ONLY, WHY_BOOK_ORIG, CURATED_ORIG))


def main():
    results = []

    for exp in experiments:
        label, explore, why_book, curated = exp
        page_text = build_page(explore, why_book, curated)
        result = call_nlp_classify(page_text)
        guided, sightsee, all_cats = extract_scores(result)
        results.append((label, guided, sightsee, all_cats))
        print(f"  {label:<55} G:{guided*100:>5.1f}%  S:{sightsee*100:>5.1f}%  gap:{(guided-sightsee)*100:>+5.1f}")
        time.sleep(0.8)

    # Summary
    print(f"\n{'='*100}")
    print(f"  ROUND 3 SUMMARY")
    print(f"{'='*100}")
    print(f"\n  {'Experiment':<60} {'Guided':>7} {'Sight':>7} {'Gap':>7}")
    print(f"  {'-'*85}")

    for label, guided, sightsee, _ in results:
        print(f"  {label:<60} {guided*100:>6.1f}% {sightsee*100:>6.1f}% {(guided-sightsee)*100:>+6.1f}")

    # Top 3 by lowest sightseeing
    print(f"\n  TOP 3 by lowest Sightseeing (with Guided preserved):")
    sorted_by_s = sorted(results, key=lambda x: x[2])
    for label, guided, sightsee, _ in sorted_by_s[:5]:
        print(f"    {sightsee*100:>5.1f}% sight | {guided*100:>5.1f}% guided | {label}")

    # Show all categories for best result
    best = sorted_by_s[0]
    print(f"\n  Full category breakdown for best result ({best[0]}):")
    for path, conf in sorted(best[3].items(), key=lambda x: -x[1]):
        print(f"    {path:<60} {conf*100:>6.1f}%")


if __name__ == "__main__":
    main()
