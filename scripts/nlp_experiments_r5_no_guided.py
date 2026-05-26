#!/usr/bin/env python3
"""
Round 5: What works WITHOUT 'guided' in the variable sections.
Product won't allow "guided" in headings/copy because "a day tour is guided too."
Find the best disambiguators that bypass this constraint.
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


# Static parts -- NOTE: "guided" appears in FAQ, H1, Curated, Top Partners, and tour cards.
# Those are EXISTING and presumably Product-approved. We're only avoiding NEW "guided" usage
# in the two variable sections.

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

# "Guided" already in these (pre-approved by Product):
HERO = "Find Guided Tours Wherever You Want to Go"
TOP_PARTNERS = "Top Tour Partners. Multi-day guided and escorted tours."
CURATED_COMPARE = "Curated Guided Tour Packages. Compare our curated collections of guided vacations, organized by tour style and travel interest."


def page(explore, why_book):
    return "\n\n".join([HERO, TOP_PARTNERS, explore, why_book, CURATED_COMPARE, CARDS, FAQ,
                        "Vacation Packages. Flights. Hotels. Car Rentals."])


experiments = []

# --- BASELINES ---

# V4 original
experiments.append(("V4-ORIG (reference)", page(
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

# R3 winner (uses "guided" -- benchmark to beat)
experiments.append(("R3-WINNER (uses guided -- benchmark)", page(
    """Find Your Guided Tour Style.
Escorted bus tours, luxury guided vacations, and small-group adventures -- match the trip to how you travel.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Guided Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour operators to build itineraries that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on guided tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European tour itineraries or planning escorted vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and guided multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour operators to match you with the right itinerary every time."""
)))


# --- NO-GUIDED EXPERIMENTS ---

# NG1: Just drop "Journeys" + "Explore", minimal other changes
experiments.append(("NG1: 'Tours by Style' + minimal WhyBook fixes", page(
    """Tours by Style.
Whether you prefer a relaxed escorted bus tour, a luxury vacation, or a small-group adventure, AMA Travel has something for everyone.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Multi-Day Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading travel tour companies to bring Albertans offers that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're looking for week-long Europe tour packages or luxury escorted vacations across Asia or Africa, you can count on AMA's expertise to help you book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and multi day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour providers to match you with the right trip every time."""
)))

# NG2: "Find Your Tour Style" (winner verb, no "guided")
experiments.append(("NG2: 'Find Your Tour Style' + dest/provider fixes", page(
    """Find Your Tour Style.
Escorted bus tours, luxury vacations, and small-group adventures -- match the trip to how you travel.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Multi-Day Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour companies to offer tours that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European tour packages or planning escorted vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour companies to match you with the right trip every time."""
)))

# NG3: Lean into "escorted" instead -- Product can't argue day tours are "escorted"
experiments.append(("NG3: lean into 'escorted' as disambiguator", page(
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

# NG4: "Organized" as disambiguator
experiments.append(("NG4: 'organized tours' framing", page(
    """Find Your Tour Style.
Escorted bus tours, organized vacations, and small-group adventures -- match the trip to how you travel.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Organized Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour companies to offer organized tours that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on organized tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European tour packages or planning an organized vacation across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and organized multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour companies to match you with the right organized tour every time."""
)))

# NG5: "Package tour" / "tour package" heavy framing
experiments.append(("NG5: 'tour packages' as primary framing", page(
    """Find Your Tour Package.
Escorted bus tours, luxury vacations, and small-group adventures -- match the tour package to how you travel.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour package with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Tour Packages with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour companies to offer tour packages that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European tour packages or planning escorted vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and multi-day tour packages tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour companies to match you with the right tour package every time."""
)))

# NG6: Best non-guided combo -- take best elements from above
# "Find Your Tour Style" + dest fix + tighter copy + "escorted" where natural
experiments.append(("NG6: best no-guided combo attempt", page(
    """Find Your Tour Style.
Escorted bus tours, luxury vacations, and small-group adventures -- match the trip to how you travel.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents are able to book everything individually for you, making the perfect tour with your interests solely in mind.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Multi-Day Tours with AMA Travel.
When you choose AMA Travel, you don't just get a multi-day tour package; you get a trusted travel partner. We work with leading tour companies to offer escorted and multi-day tours that are safe, reliable, and memorable.
As an AMA member, you'll enjoy exclusive perks like discounts on tour packages, savings on travel medical insurance, and tips from our travel experts. Whether you're comparing European tour packages or planning escorted vacations across Asia or Africa, count on AMA's expertise to book with confidence. Match with a Travel Agent.
Endless Options. Choose from a wide selection of escorted, hosted, and multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour companies to match you with the right trip every time."""
)))

# NG7: Same as NG6 but with "travel style" consistency + trimmed "luxury vacations"
experiments.append(("NG7: NG6 + tighter, 'travel style' consistent", page(
    """Find Your Tour Style.
Escorted bus tours, luxury vacations, and small-group adventures -- find the tour that fits your travel style.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents build a custom tour around your interests, booking each detail individually.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Multi-Day Tours with AMA Travel.
Booking with AMA Travel means a trusted partner from first call to final day. We work with leading tour companies to offer escorted and multi-day tours that are safe, reliable, and built for comfort.
AMA members enjoy exclusive perks: discounts on tour packages, savings on travel medical insurance, and direct access to our Alberta-based travel experts. Comparing European tour packages or planning an escorted vacation through Asia? Our agents match you with the right tour company and trip. Match with a Travel Agent.
Endless Options. Escorted, hosted, and multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour companies to find you the right tour every time."""
)))

# NG8: What if we can sneak ONE "guided" into the subtext? Just the explore subtext.
experiments.append(("NG8: ONE sneaky 'guided' in explore subtext only", page(
    """Find Your Tour Style.
Escorted bus tours, luxury guided vacations, and small-group adventures -- find the tour that fits your travel style.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents build a custom tour around your interests, booking each detail individually.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Multi-Day Tours with AMA Travel.
Booking with AMA Travel means a trusted partner from first call to final day. We work with leading tour companies to offer escorted and multi-day tours that are safe, reliable, and built for comfort.
AMA members enjoy exclusive perks: discounts on tour packages, savings on travel medical insurance, and direct access to our Alberta-based travel experts. Comparing European tour packages or planning an escorted vacation through Asia? Our agents match you with the right tour company and trip. Match with a Travel Agent.
Endless Options. Escorted, hosted, and multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour companies to find you the right tour every time."""
)))

# NG9: "Multi-Day Tour Styles" -- keeps "multi-day" (Product happy) but drops "Journeys"
experiments.append(("NG9: 'Multi-Day Tour Styles' heading", page(
    """Multi-Day Tour Styles.
Escorted bus tours, luxury vacations, and small-group adventures -- find the tour that fits your travel style.
Bus Tours. Solo Tours. Adventure Tours. Escorted Tours.
Custom Tour. Our expert agents build a custom tour around your interests, booking each detail individually.""",
    """Plan Your Perfect Tour with Travel Experts.
Why Book Multi-Day Tours with AMA Travel.
Booking with AMA Travel means a trusted partner from first call to final day. We work with leading tour companies to offer escorted and multi-day tours that are safe, reliable, and built for comfort.
AMA members enjoy exclusive perks: discounts on tour packages, savings on travel medical insurance, and direct access to our Alberta-based travel experts. Comparing European tour packages or planning an escorted vacation through Asia? Our agents match you with the right tour company and trip. Match with a Travel Agent.
Endless Options. Escorted, hosted, and multi-day tours tailored to your travel style.
Expert Advice. Our Alberta-based travel agents work with top tour companies to find you the right tour every time."""
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

    print(f"\n{'='*100}")
    print(f"  NO-GUIDED RESULTS")
    print(f"{'='*100}")
    print(f"\n  {'Experiment':<60} {'Guided':>7} {'Sight':>7} {'Gap':>7} {'vs V4':>7}")
    print(f"  {'-'*88}")

    for label, g, s, _ in results:
        vs_v4 = (s - orig_s) * 100
        print(f"  {label:<60} {g*100:>6.1f}% {s*100:>6.1f}% {(g-s)*100:>+6.1f} {vs_v4:>+6.1f}")

    # Best no-guided options (skip first two baselines)
    no_guided = [(l, g, s) for l, g, s, _ in results[2:]]
    print(f"\n  BEST NO-GUIDED OPTIONS (sorted by lowest Sightseeing):")
    for label, g, s in sorted(no_guided, key=lambda x: x[2]):
        vs_v4 = (s - orig_s) * 100
        vs_winner = (s - results[1][2]) * 100
        print(f"    S:{s*100:>5.1f}%  G:{g*100:>5.1f}%  vs-V4:{vs_v4:>+5.1f}pp  vs-guided-winner:{vs_winner:>+5.1f}pp  | {label}")


if __name__ == "__main__":
    main()
