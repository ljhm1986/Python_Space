import pandas as pd
import numpy as np 

#3등 이내 2점, 2개 이상 1점, 그외 0점 

#2013
goty2013A = ['The Last of Us', 'Grand Theft Auto V', 'BioShock Infinite']
goty2013B= ['Super Mario 3D World', 'Gone Home', 'Tome Raider',
 'The Legend of Zelda: A Link Between Worlds',
 "Assassin's Creed IV: Black Flag", 'Papers, Please', 
 'Rayman Legends', 'Brothers: A Tale of Two Sons', 'The Stanley Parable',
 'Pokémon X·Y', 'Battlefield 4', 'Ni no Kuni: Wrath of the White Witch']
#2014
goty2014A= ['Dragon Age: Inquisition', 'Middle-earth: Shadow of Mordor',
'Mario Kart 8', 'Super Smash Bros.']#공동 3등
goty2014B= ['Far Cry 4','Alien: Isolation','Bayonetta 2', 'Dark Souls II', 'Destiny',
 'Hearthstone: Heroes of Warcraft', 'Kentucky Route Zero - Act III',
 'Sunset Overdrive', 'Divinity: Original Sin', 'Forza Horizon 2', 
 'Wolfenstein: The New Order', 'This War of Mine', 'Titanfall',
 'Call of Duty: Advanced Warfare','Danganronpa: Trigger Happy Havoc',
 'Shovel Knight', 'South Park: The Stick of Truth', 'The Evil Within', 
 'The Walking Dead: Season Two', 'Transistor']
#2015
goty2015A= ['The Witcher 3: Wild Hunt', 'Fallout 4', 'Bloodborne']
goty2015B= ['Metal Gear Solid V: The Phantom Pain', 'Life Is Strange', 'Super Mario Maker',
 'Undertale', 'Rocket League', 'Rise of the Tomb Raider', 'Batman: Arkham Knight',
 'Her Story', "Assassin's Creed Syndicate", 'Splatoon']
#2016
goty2016A= ["Uncharted 4: A Thief's End", 'Overwatch', 'DOOM']
goty2016B= ['Battlefield 1', 'The Last Guardian', 'INSIDE', 'Dark Souls III','Final Fantasy XV',
 'The Witcher 3: Wild Hunt - Blood and Wine', 'Dishonored 2', 'The Witness',
 'Titanfall 2', 'Hitman', 'Pokémon GO', 'XCOM 2', 'SUPERHOT']
#2017
goty2017A= ['The Legend of Zelda: Breath of the Wild', 'Horizon Zero Dawn', 'Super Mario Odyssey']
goty2017B= ['NieR: Automata', 'Persona 5', "PlayerUnknown's Battlegrounds",
 "Assassin's Creed Origins", 'Divinity: Original Sin II', 'Wolfenstein II: The New Colossus',
 'Prey', 'Resident Evil 7: Biohazard', "Hellblade: Senua's Sacrifice"]
#2018
goty2018A= ['God of War', 'Red Dead Redemption 2', "Marvel's Spider-Man"]
goty2018B= ['Celeste', 'Return of the Obra Dinn', 'Fortnite Battle Royale', 'Monster Hunter: World',
 'Super Smash Bros. Ultimate', 'Tetris Effect', 'Astro Bot Rescue Mission',
 'Florence', 'Kingdom Come: Deliverance']

gotyA = []
gotyA.extend(goty2013A)

count = 0
for i in metascoreDF['name']:
    if i in goty2013A:
        count += 1

count = 0
for i in opencriticDF['name']:
    if i in goty2013A:
        count += 1

opencriticDF['name'][opencriticDF['openscore'] >= 90]

