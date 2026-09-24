"""Football data providers behind one provider-independent interface.

`base.FootballDataProvider` is the contract, `schemas` holds the neutral record
types, and each subpackage (e.g. `api_football`) adapts one external API to it.
"""
