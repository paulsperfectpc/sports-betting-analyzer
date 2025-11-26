### New scope  
This is a new Project File for v2. Please modify any required changes to existing project files 

## New API - SportsData.io 
https://sportsdata.io/developers/api-documentation/nfl
https://sportsdata.io/developers/api-documentation/nba
https://sportsdata.io/developers/api-documentation/nhl

1000 pulls per month.  

API KEY: 1fdd78185de84dc1bd82ff59f254c087


### Scrap project and start fresh
Utilizing SportsData.io API for ALL team/player historical history along with betting lines. 

For Team betting lines, pull the Over/Under for score, spread and moneyline. Player props should only pull player data when a user requests player props. When a user searches for a team OR a player, the last 10 games should be displayed. need to be mindful and effienct of API limitations. 

The app should still work the same - end user searches for 1 or 2 teams, when a user searches for 1 team, only the last 10 matches of that team should be displayed along with the outcome. If a user searches for 2 teams, the last 5 games for each team should be displayed along with any upcoming betting lines for the teams (spread, total points, moneyline, etc). When a User searches for a player (NFL, NBA, NHL) the last 10 games should be displayed, if there are any betting lines for the player for that day, those betting lines should be displayed. 

Once all of the data is loaded (player or team), the data then needs to get fed into the local LLM (qwen3:4b) for analysis of the betting lines. The AI needs to be instructed to determine the MOST likely player prop and/or team betting lines are most likely to cover (spread, total points, winner)


I also want to STORE and save any data that is looked up. For instance, If I look up 2 teams, instead of reaching out to the API for ALL new data, only use the API to call for NEW data (show older 9 games, pull data for 1 new game) cache and SAVE ALL lookup and historical data following the parameters above. the same logic should be applied to players as well. Save ALL lookup data. the idea being, the LLM should be able to recall the conversation history and provide a more accurate analysis of future games/matchups and betting props. 


Hosting requirements are the same: on PhotonOS machine running portainer @ 10.254.254.220 