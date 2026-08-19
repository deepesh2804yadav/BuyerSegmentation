"""Named buyer-segment copy used by clustering and the dashboard."""

SEGMENT_PLAYBOOK = {
    "C1": {
        "buyer_type": "Global Investors",
        "characteristics": "Highest share of non-US buyers and solid investment-purpose demand.",
        "marketing": "Target with yield, FX-aware pricing, and cross-border investment packs.",
    },
    "C2": {
        "buyer_type": "First-Time Buyers",
        "characteristics": "Largest owner-occupier pool: home-led purchases with frequent loan use.",
        "marketing": "Lead with financing partners, starter-to-mid inventory, and education content.",
    },
    "C3": {
        "buyer_type": "Corporate Buyers",
        "characteristics": "Registered company accounts. Younger decision-makers buying multiple units.",
        "marketing": "Offer bulk pricing, office mix, and relationship-managed deals.",
    },
    "C4": {
        "buyer_type": "Luxury Investors",
        "characteristics": "Small high-value cohort: older buyers, higher satisfaction, largest portfolios.",
        "marketing": "Prioritize concierge sales, premium towers, and exclusive listings.",
    },
}
