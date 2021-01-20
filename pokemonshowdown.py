# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer

from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from poke_env.player.player import Player

from tensorflow import keras

from poke_env.player.baselines import SimpleHeuristicsPlayer

from poke_env.environment.pokemon_type import PokemonType
import asyncio


# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Player):
	model = None

	def embed_battle(self, battle):
		# -1 indicates that the move does not have a base power
		# or is not available
		estat = battle.opponent_active_pokemon.base_stats

		mystat = battle.active_pokemon.base_stats

		my_pokemon = []
		#create stuff about my pokemon
		for pokemon in battle.team.values():
			#do something
			current_mon = []
			#embed HP
			current_mon.append(pokemon.current_hp)
			current_mon.append(pokemon.current_hp_fraction)
			#Embed moves
			for move in pokemon.moves:
				movestats = []
				movestats.append(move.accuracy)
				movestats.append(move.base_power)
				movestats.append(move.crit_ratio)
				movestats.append(move.current_pp)
				movestats.append(move.expected_hits)
				movestats.append(move.type)

		their_pokemon = []
		for pokemon in battle.opponent_team.values():


		damages = [0, 0, 0, 0]

		for i in range(len(battle.available_moves)):
				damages[i] = damageCalc(battle.active_pokemon,battle.opponent_active_pokemon, battle.available_moves[i], battle.weather, battle.fields)

		# We count how many pokemons have not fainted in each team
		remaining_mon_team = (
			len([mon for mon in battle.team.values() if mon.fainted]) / 6
		)
		remaining_mon_opponent = (
			len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
		)

		# Final vector with 10 components
		return np.concatenate(
			[
				damages
				[remaining_mon_team, remaining_mon_opponent],
				[int(battle.trapped)]
			]
		)

	def damageCalc(my_poke, opp_poke, move,weather,terrain):

		damage = move.base_power

		# takes into account skill link
		if move.expected_hits>2:
			damage*=5
		else:
			damage*=move.expected_hits

		# Accounts for abilities that make pokemon immune to certain attacks
		for j in opp_poke.possible_abilities:
			i = opp_poke.possible_abilities[j]

			if i == "Levitate" and move.type==PokemonType.GROUND:
				damage*=0
			if i == "Flash Fire"and move.type==PokemonType.FIRE:
				damage*=0

			if i == "Lightning Rod"and move.type==PokemonType.ELECTRIC:
				damage*=0

			if i == "Motor Drive"and move.type==PokemonType.ELECTRIC:
				damage*=0

			if i == "Volt Absorb"and move.type==PokemonType.ELECTRIC:
				damage*=0

			if i == "Water Absorb"and move.type==PokemonType.WATER:
				damage*=0

			if i == "Dry Skin"and move.type==PokemonType.WATER:
				damage*=0

			if i == "Sap Sipper"and move.type==PokemonType.GRASS:
				damage*=0

			if i == "Storm Drain"and move.type==PokemonType.WATER:
				damage*=0

		# accounts for STAB
		for i in my_poke.types:
			if i == move.type:
				damage *= 1.5

		# accounts for effectiveness
		damage *= move.type.damage_multiplier(opp_poke.type_1, opp_poke.type_2)

		# accounts for weather
		if weather != None:
			if weather.RAINDANCE == weather:
				if move.type == PokemonType.WATER:
					damage*=1.5

				if move.type == PokemonType.FIRE:
					damage *=.677777


			if weather.SUNNYDAY == weather:
				if move.type == PokemonType.FIRE:
					damage *= 1.5
				if move.type == PokemonType.WATER:
					damage *= .677777

		# accounts for terrain
		for field in terrain:
			if field == field.ELECTRIC_TERRAIN:
				if move.type == PokemonType.ELECTRIC:
					damage *= 1.3

			if field == field.MISTY_TERRAIN:
				if move.type == PokemonType.DRAGON:
					damage *= .5

			if field == field.PSYCHIC_TERRAIN:
				if move.type == PokemonType.PSYCHIC:
					damage*= 1.3
			if field == field.GRASSY_TERRAIN:
				if move.type == PokemonType.GRASS:
					damage*= 1.3
				if move.type == PokemonType.GROUND:
					damage*=.5

		return damage


	def choose_move(self, battle):
		# If the player can attack, it will7
		observations = self.embed_battle(battle)
		#print(observations.shape)
		
		observations = observations.reshape(1, 1, 11)
		actionArry = self.model.predict(observations)
		print(actionArry)
		action = np.argmax(actionArry)
		print(action)

		if (
			action < 4
			and action < len(battle.available_moves)
			and not battle.force_switch
		):
			return self.create_order(battle.available_moves[action])
		elif (
			not battle.force_switch
			and battle.can_z_move
			and battle.active_pokemon
			and 0
			<= action - 4
			< len(battle.active_pokemon.available_z_moves)  # pyre-ignore
		):
			return self.create_order(
				battle.active_pokemon.available_z_moves[action - 4], z_move=True
			)
		elif (
			battle.can_mega_evolve
			and 0 <= action - 8 < len(battle.available_moves)
			and not battle.force_switch
		):
			return self.create_order(battle.available_moves[action - 8], mega=True)
		elif (
			battle.can_dynamax
			and 0 <= action - 12 < len(battle.available_moves)
			and not battle.force_switch
		):
			return self.create_order(battle.available_moves[action - 12], dynamax=True)
		elif 0 <= action - 16 < len(battle.available_switches):
			return self.create_order(battle.available_switches[action - 16])
		else:
			return self.choose_random_move(battle)



async def main():
	# player = SimpleRLPlayer(
	#     battle_format="gen8randombattle",
	#     player_configuration=PlayerConfiguration("MrMonacle", "ac30022me"),
	#     server_configuration=ShowdownServerConfiguration,
	# )
	# player.model = keras.models.load_model('model_30000_vsMax')

	# player1 = RandomPlayer(
	#     battle_format="gen8randombattle",
	#     player_configuration=PlayerConfiguration("MrMonacle", "ac30022me"),
	#     server_configuration=ShowdownServerConfiguration,
	# )

	huerPlayer = SimpleHeuristicsPlayer(
		battle_format="gen8randombattle",
		player_configuration=PlayerConfiguration("MrMonacle", "ac30022me"),
		server_configuration=ShowdownServerConfiguration,
		start_timer_on_battle_start=True,
	)

	


	await huerPlayer.ladder(5)

if __name__ == "__main__":
	asyncio.get_event_loop().run_until_complete(main())



