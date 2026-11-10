# FPL prediction model (Windows)

Predicts Fantasy Premier League points for every player, picks your best XI, captain and transfers,
and writes one offline dashboard page (out\dashboard.html). Runs only on your computer. $0.

## Files
- fpl_model.py            the model (data -> expected points -> simulations -> optimiser -> dashboard)
- dashboard_template.html dashboard layout; data is embedded at run time
- requirements.txt        exact Python library versions
- my_team.txt             YOUR squad, bank, free transfers, selling prices (plain text, never executed, not in git)
- my_team.example.txt     template committed to git
- setup.bat               one-time setup: creates .venv and installs the libraries
- update.bat              weekly: refresh data, run the model, open the dashboard ("update.bat auto" = no pause)
- player.bat              one player in the terminal: player.bat Haaland

## Every gameweek
Edit my_team.txt after transfers, then double-click update.bat (after the gameweek ends, and on deadline morning).
If the FPL site doesn't answer, it retries with the GitHub copy of the data automatically.

## Security
- Talks only to fantasy.premierleague.com and raw.githubusercontent.com (anything else is refused).
- No login or password. The dashboard makes no network requests (CSP connect-src 'none').
