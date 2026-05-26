#!/usr/bin/env python3
"""
Round 4: Final confirmation + sensitivity tests on the winner.
"""

import json
import time
import requests

BRIDGE_URL = "https://ghzfrxxevjjfgpxvmahy.supabase.co/functions/v1/bridge"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdoemZyeHhldmpqZmdweHZtYWh5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgxNzk4OTgsImV4cCI6MjA5Mzc1NTg5OH0.VPJTvdCvU217QmQBjm3ym8ZOoCgyBY-VpLpPdhefa04"


def call_nlp(text):
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


def scores(result):
    g, s = 0, 0
    cats = {}
    for c in result.get("categories", []):
        name = c["name"].strip().lstrip("/")
        conf = c["confidence"]
        cats[name] = conf
        if "guided tours" in name.lower() or "escorted" in name.lower():
            g = max(g, conf)
        if "sightseeing" in name.lower():
            s = max(s, conf)
    return g, s, cats


# Static parts
FAQ = """Guided Tour FAQs.
What is the definition of a tour? A tour is a structured trip involving a planned itinerary, typically organized and led by a professional guide or tour company.
What is the difference between escorted and guided tours? Escorted tours provide a dedicated tour director who travels with the group throughout the entire trip, while guided tours may use local guides at specific stops or attractions.
What is the best tour company in Canada? Some of the top tour companies operating in Canada include Trafalgar, Insight Vacations, Collette, and G Adventures. AMA Travel partners with leading tour operators to bring Albertans a curated selection of guided and escorted tour options.
Is it cheaper to travel with a tour company? In many cases, yes. Tour companies negotiate group rates on hotels, transport, and attractions, passing savings on to travellers. Packages often include meals, transfers, and tips, making it easier to budget.
What is the best site for tour packages? AMA Travel offers a wide selection of multi-day tour packages from trusted tour operators. Our Alberta-based travel agents help you compare options and find the right fit.
What is a multi-day tour? A multi-day tour is an organized trip lasting two or more days that follows a set itinerary, with accommodation, transportation, and guided activities included."""

CARDS = """Trafalgar European Discovery 14 days from $3,299 per person.
Insight Vacations Country Roads of Italy 10 days from $4,195 per person.
Collette Exploring South Africa 14 days from $5,099 per person.
G Adventures Costa Rica Quest 9 days from $1,799 per person.
Globus Hawaiian Adventure 10 days from $3,599 per person.
Contiki European Discovery 14 days from $2,499 per person."""


def page(hero, top, explore, why_book, curated):
    return "\n\n".join([hero, top, explore, why_book, curated, CARDS, FAQ, "Vacation Packages. Flights. Hotels. Car Rentals."])


experiments = []

# --- V4 ORIGINAL (from optimization guide) ---
experiments.append(("V4-ORIG", page(
    "Find Guided Tours Wherever You Want to Go",
    "Top Tour Partners. Multi-day guided and escorted tours.",
    """Explore Multi-Day Journeys by Style.
Whether you prefer a relaxed escorted bus tour, a luxury guided vacation, or a small-group adventure, AMA Travel has something for everyone.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Multi-Day Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading travel tour companies to bring Albertans offers that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on tour packages, savings on travel medical insurance, and insider tips from our destination experts. Whether you're looking for week-long Europe tour packages or luxury escorted vacations across Asia or Africa, you can count on AMA's expertise to help you book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and guided multi day tours tailored to your style and interests.
Expert Advice. Our Alberta-based travel agents work with top tour providers to match you with the right trip every time.""",
    "Curated Guided Tour Packages. Browse our curated collections of guided vacations, grouped by style and interest."
)))

# --- R3 WINNER (T1) ---
experiments.append(("R3-WINNER", page(
    "Find Guided Tours Wherever You Want to Go",
    "Top Tour Partners. Multi-day guided and escorted tours.",
    """Find Your Guided Tour Style.
Escorted bus tours, luxury guided vacations, and small-group adventures -- match the trip to how you travel.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Guided Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour operators to build itineraries that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on guided tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European tour itineraries or planning escorted vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and guided multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour operators to match you with the right itinerary every time.""",
    "Curated Guided Tour Packages. Compare our curated collections of guided vacations, organized by tour style and travel interest."
)))

# --- SENSITIVITY: what if UX can't do "Find Your"? Back to label-only ---
experiments.append(("SENS-1: label-only heading 'Guided Tour Styles'", page(
    "Find Guided Tours Wherever You Want to Go",
    "Top Tour Partners. Multi-day guided and escorted tours.",
    """Guided Tour Styles.
Escorted bus tours, luxury guided vacations, and small-group adventures -- match the trip to how you travel.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Guided Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour operators to build itineraries that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on guided tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European tour itineraries or planning escorted vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and guided multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour operators to match you with the right itinerary every time.""",
    "Curated Guided Tour Packages. Compare our curated collections of guided vacations, organized by tour style and travel interest."
)))

# --- SENSITIVITY: what if UX insists on keeping "Explore"? Pair with Guided ---
experiments.append(("SENS-2: 'Explore Guided Tour Styles'", page(
    "Find Guided Tours Wherever You Want to Go",
    "Top Tour Partners. Multi-day guided and escorted tours.",
    """Explore Guided Tour Styles.
Escorted bus tours, luxury guided vacations, and small-group adventures -- match the trip to how you travel.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Guided Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour operators to build itineraries that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on guided tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European tour itineraries or planning escorted vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and guided multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour operators to match you with the right itinerary every time.""",
    "Curated Guided Tour Packages. Compare our curated collections of guided vacations, organized by tour style and travel interest."
)))

# --- SENSITIVITY: WhyBook MUST keep "Multi-Day" in H3? ---
experiments.append(("SENS-3: forced 'Multi-Day' in WhyBook H3", page(
    "Find Guided Tours Wherever You Want to Go",
    "Top Tour Partners. Multi-day guided and escorted tours.",
    """Find Your Guided Tour Style.
Escorted bus tours, luxury guided vacations, and small-group adventures -- match the trip to how you travel.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Multi-Day Guided Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour operators to build itineraries that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on guided tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European tour itineraries or planning escorted vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and guided multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour operators to match you with the right itinerary every time.""",
    "Curated Guided Tour Packages. Compare our curated collections of guided vacations, organized by tour style and travel interest."
)))

# --- BONUS: what does dropping "adventures" from subtext do? ---
experiments.append(("BONUS-1: subtext 'hosted group tours' not 'adventures'", page(
    "Find Guided Tours Wherever You Want to Go",
    "Top Tour Partners. Multi-day guided and escorted tours.",
    """Find Your Guided Tour Style.
Escorted bus tours, luxury guided vacations, and hosted group tours -- match the trip to how you travel.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Guided Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour operators to build itineraries that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on guided tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European tour itineraries or planning escorted vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and guided multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour operators to match you with the right itinerary every time.""",
    "Curated Guided Tour Packages. Compare our curated collections of guided vacations, organized by tour style and travel interest."
)))

# --- BONUS: what about "Match Your" instead of "Find Your"? ---
experiments.append(("BONUS-2: 'Match Your Guided Tour Style'", page(
    "Find Guided Tours Wherever You Want to Go",
    "Top Tour Partners. Multi-day guided and escorted tours.",
    """Match Your Guided Tour Style.
Escorted bus tours, luxury guided vacations, and small-group adventures -- find the trip that fits how you travel.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Guided Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour operators to build itineraries that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on guided tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European tour itineraries or planning escorted vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and guided multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour operators to match you with the right itinerary every time.""",
    "Curated Guided Tour Packages. Compare our curated collections of guided vacations, organized by tour style and travel interest."
)))


def main():
    results = []
    for label, text in experiments:
        result = call_nlp(text)
        g, s, cats = scores(result)
        results.append((label, g, s, cats))
        print(f"  {label:<55} G:{g*100:>5.1f}%  S:{s*100:>5.1f}%  gap:{(g-s)*100:>+5.1f}")
        time.sleep(0.8)

    orig_s = results[0][2]
    winner_s = results[1][2]

    print(f"\n{'='*100}")
    print(f"  FINAL RESULTS")
    print(f"{'='*100}")
    print(f"\n  {'Experiment':<55} {'Guided':>7} {'Sight':>7} {'Gap':>7} {'vs V4':>7}")
    print(f"  {'-'*80}")

    for label, g, s, _ in results:
        vs_v4 = (s - orig_s) * 100
        print(f"  {label:<55} {g*100:>6.1f}% {s*100:>6.1f}% {(g-s)*100:>+6.1f} {vs_v4:>+6.1f}")

    print(f"\n  V4-ORIG Sightseeing:  {orig_s*100:.1f}%")
    print(f"  R3-WINNER Sightseeing: {winner_s*100:.1f}%")
    print(f"  Reduction:             {(orig_s-winner_s)*100:+.1f}pp")


if __name__ == "__main__":
    main()
