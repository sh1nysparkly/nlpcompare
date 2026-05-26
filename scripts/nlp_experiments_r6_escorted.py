#!/usr/bin/env python3
"""
Round 6: Escorted framing deep-dive.
NG3 beat the guided winner. Confirm and tune.
Also test whether "guided vacation" (not "guided tour") survives Product objections.
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

HERO = "Find Guided Tours Wherever You Want to Go"
TOP_PARTNERS = "Top Tour Partners. Multi-day guided and escorted tours."
CURATED_COMPARE = "Curated Guided Tour Packages. Compare our curated collections of guided vacations, organized by tour style and travel interest."


def page(explore, why_book):
    return "\n\n".join([HERO, TOP_PARTNERS, explore, why_book, CURATED_COMPARE, CARDS, FAQ,
                        "Vacation Packages. Flights. Hotels. Car Rentals."])


experiments = []

# --- REFERENCE POINTS ---
experiments.append(("V4-ORIG", page(
    """Explore Multi-Day Journeys by Style.
Whether you prefer a relaxed escorted bus tour, a luxury guided vacation, or a small-group adventure, AMA Travel has something for everyone.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Multi-Day Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading travel tour companies to bring Albertans offers that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on tour packages, savings on travel medical insurance, and insider tips from our destination experts. Whether you're looking for week-long Europe tour packages or luxury escorted vacations across Asia or Africa, you can count on AMA's expertise to help you book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and guided multi day tours tailored to your style and interests.
Expert Advice. Our Alberta-based travel agents work with top tour providers to match you with the right trip every time."""
)))

# NG3 exact (confirm)
experiments.append(("NG3 confirm (escorted heavy)", page(
    """Find Your Tour Style.
Escorted tours, luxury vacations, and small-group adventures -- match the trip to how you travel.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Escorted and Multi-Day Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour companies to offer escorted tours and vacation packages that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on escorted tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European escorted tours or planning vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour companies to match you with the right escorted tour every time."""
)))

# --- TUNING: can we keep Guided higher while keeping the escorted sightseeing win? ---

# E1: Mix -- "escorted" in H3 + body but keep "guided vacation" in explore subtext
experiments.append(("E1: escorted WhyBook + 'guided vacations' in explore", page(
    """Find Your Tour Style.
Escorted tours, luxury guided vacations, and small-group adventures -- match the trip to how you travel.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Escorted and Multi-Day Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour companies to offer escorted tours and vacation packages that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on escorted tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European escorted tours or planning vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour companies to match you with the right escorted tour every time."""
)))

# E2: Lighter escorted touch -- only in H3 and value props, not saturating body
experiments.append(("E2: light escorted -- H3 + value props only", page(
    """Find Your Tour Style.
Escorted bus tours, luxury guided vacations, and small-group adventures -- match the trip to how you travel.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Escorted and Multi-Day Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour companies to offer tours and vacation packages that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European tour packages or planning escorted vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour companies to match you with the right tour every time."""
)))

# E3: "Escorted" H3 but body uses "guided vacation" and "escorted" in balance
experiments.append(("E3: balanced -- 'escorted' H3, mix guided/escorted body", page(
    """Find Your Tour Style.
Escorted bus tours, luxury guided vacations, and small-group adventures -- match the trip to how you travel.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Escorted and Multi-Day Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour companies to offer escorted tours and guided vacations that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European escorted tours or planning guided vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour companies to match you with the right tour every time."""
)))

# E4: What if we just change the H3 to "Why Book Escorted Tours" (not "Escorted and Multi-Day")?
experiments.append(("E4: H3 = 'Why Book Escorted Tours with AMA Travel'", page(
    """Find Your Tour Style.
Escorted bus tours, luxury guided vacations, and small-group adventures -- match the trip to how you travel.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Escorted Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour companies to offer escorted tours and guided vacations that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European escorted tours or planning guided vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour companies to match you with the right tour every time."""
)))

# E5: Keep WhyBook H3 as "Multi-Day Tours" but use escorted framing in body only
experiments.append(("E5: orig WhyBook H3 + escorted body framing", page(
    """Find Your Tour Style.
Escorted bus tours, luxury guided vacations, and small-group adventures -- match the trip to how you travel.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Multi-Day Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour companies to offer escorted tours and guided vacations that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on escorted tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European escorted tours or planning guided vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour companies to match you with the right tour every time."""
)))

# E6: "Find Your Tour Style" + MINIMAL WhyBook changes (only dest->travel experts + escorted body)
# This is the "least resistance" version -- smallest diff from V4
experiments.append(("E6: minimum viable -- heading fix + dest fix + light escorted", page(
    """Find Your Tour Style.
Whether you prefer a relaxed escorted bus tour, a luxury guided vacation, or a small-group adventure, AMA Travel has something for everyone.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Multi-Day Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading travel tour companies to bring Albertans escorted tours and guided vacations that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're looking for week-long European escorted tours or luxury guided vacations across Asia or Africa, you can count on AMA's expertise to help you book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and guided multi day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour providers to match you with the right trip every time."""
)))


def main():
    results = []
    for label, text in experiments:
        result = call_nlp(text)
        g, s, cats = scores(result)
        results.append((label, g, s, cats))
        print(f"  {label:<60} G:{g*100:>5.1f}%  S:{s*100:>5.1f}%  gap:{(g-s)*100:>+5.1f}")
        time.sleep(0.8)

    orig_s = results[0][2]
    orig_g = results[0][1]

    print(f"\n{'='*100}")
    print(f"  ESCORTED DEEP-DIVE RESULTS")
    print(f"{'='*100}")
    print(f"\n  {'Experiment':<60} {'Guided':>7} {'Sight':>7} {'Gap':>7} {'S vs V4':>7} {'G vs V4':>7}")
    print(f"  {'-'*100}")

    for label, g, s, _ in results:
        vs_v4_s = (s - orig_s) * 100
        vs_v4_g = (g - orig_g) * 100
        marker = " <-- BEST" if s == min(r[2] for r in results) else ""
        print(f"  {label:<60} {g*100:>6.1f}% {s*100:>6.1f}% {(g-s)*100:>+6.1f} {vs_v4_s:>+6.1f} {vs_v4_g:>+6.1f}{marker}")

    print(f"\n  Full categories for top 3 by lowest Sightseeing:")
    sorted_results = sorted(results, key=lambda x: x[2])
    for label, g, s, cats in sorted_results[:3]:
        print(f"\n    {label}  (G:{g*100:.1f}% S:{s*100:.1f}%)")
        for path, conf in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"      {path:<60} {conf*100:>6.1f}%")


if __name__ == "__main__":
    main()
