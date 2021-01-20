# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from poke_env.player.player import Player

from poke_env.environment.pokemon_type import PokemonType

from poke_env.player.baselines import SimpleHeuristicsPlayer

from poke_env.player.random_player import RandomPlayer
from poke_env.player.player import Player
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.pokemon_type import PokemonType
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration
from poke_env.environment.move import Move
from poke_env.environment.status import Status

def switch(player, battle):
	index2 = 0
	max2 = 0

	# Switches to a pokemon with an effective typing
	for i in range(len(battle.available_switches)):
		if battle.opponent_active_pokemon.damage_multiplier(battle.available_switches[i].type_1) > max2:
			index2 = i
			max2 = battle.opponent_active_pokemon.damage_multiplier(battle.available_switches[i].type_1)
		if battle.opponent_active_pokemon.damage_multiplier(battle.available_switches[i].type_2) > max2:
			index2 = i
			max2 = battle.opponent_active_pokemon.damage_multiplier(battle.available_switches[i].type_2)

	# If no pokemon has an effective typing, it switches into a pokemon that resists
	if max2 <= 1:
		max2 = 4
		index2 = 0
		for i in range(len(battle.available_switches)):
			if battle.available_switches[i].damage_multiplier(battle.opponent_active_pokemon.type_1) < max2:
				index2 = i
				max2 = battle.available_switches[i].damage_multiplier(battle.opponent_active_pokemon.type_1)
	if len(battle.available_switches) != 0:
		return player.create_order(battle.available_switches[index2])

def typeToInt(mtype):

	if mtype == PokemonType.BUG:
		return 0
	elif mtype == PokemonType.DARK:
		return 1
	elif mtype == PokemonType.DRAGON:
		return 2
	elif mtype == PokemonType.ELECTRIC:
		return 3
	elif mtype == PokemonType.FAIRY:
		return 4
	elif mtype == PokemonType.FIGHTING:
		return 5
	elif mtype == PokemonType.FIRE:
		return 6
	elif mtype == PokemonType.FLYING:
		return 7
	elif mtype == PokemonType.GHOST:
		return 8
	elif mtype == PokemonType.GRASS:
		return 9
	elif mtype == PokemonType.GROUND:
		return 10
	elif mtype == PokemonType.ICE:
		return 11
	elif mtype == PokemonType.NORMAL:
		return 12
	elif mtype == PokemonType.POISON:
		return 13
	elif mtype == PokemonType.PSYCHIC:
		return 14
	elif mtype == PokemonType.ROCK:
		return 15
	elif mtype == PokemonType.STEEL:
		return 16
	elif mtype == PokemonType.WATER:
		return 17
	else:
		return -1


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

class PokeBot(Player):

	# How many pokemon the opponent has
	global eteam_size

	# If the pokemon used protect last turn
	global protect_last

	def choose_move(self, battle):

		# Moves the pokemon checks for
		protect = Move("protect")
		glare = Move("glare")
		twave = Move("thunderwave")
		toxic = Move("toxic")
		wisp = Move("willowisp")
		web = Move("stickyweb")
		bbunker = Move("banefulbunker")
		spikyshield = Move("spikyshield")


		# Statuses the bot checks for
		faint1 = Status(2)
		toxic2 = Status(7)

		#stats of the active enemy pokemon
		estat = battle.opponent_active_pokemon.base_stats

		#stats of the bots active pokemon
		mystat = battle.active_pokemon.base_stats

		#used to find how many pokemon the enemy has
		self.eteam_size = 6
		for enemy in battle.opponent_team:
			if battle.opponent_team[enemy].status== 2 or battle.opponent_team[enemy].status == faint1:
				self.eteam_size-=1

		# protect_last should be made False at the start of every battle
		if battle.turn == 1:
			self.protect_last = False

		# checks if the enemy pokemon is dynamaxed
		if battle.dynamax_turns_left == None:

			# Checks if the active pokemon has certain non damaging moves and uses them if conditions are met
			for m in battle.available_moves:
			   # print(str(m)+" : " + str(m.id))

				# use sticky web if the opponent has more than 4 pokemon alive
				if m.id == web.id and self.eteam_size>=4:
					print(battle.battle_tag)
					immune = False
					# checks if opponent already has sticky web
					for condition in battle.opponent_side_conditions:
						if condition == SideCondition.STICKY_WEB:
							immune = True
					if not immune:
					 return self.create_order(m)

				# uses protect if the opponent's active pokemon is badly poisoned
				if (m.id==protect.id or m.id == spikyshield.id or m.id == bbunker.id) \
						and (battle.opponent_active_pokemon.status==toxic2 or
										 battle.opponent_active_pokemon.status==7):
					# checks if the active pokemon didn't use it last turn
					if self.protect_last == False:
						self.protect_last = True
						return self.create_order(m)
					else:
						self.protect_last = False

				# checks if the opponent's pokemon doesn't have another status
				if m.id==glare.id and battle.opponent_active_pokemon.status==None:
					immune = False
					# checks if the enemy pokemon is immune
					for type in battle.opponent_active_pokemon.types:
						if type == PokemonType.ELECTRIC:
							immune = True
					# uses glare if the opponent not immune and is faster than the active pokemon
					if not immune and estat["spe"]>=mystat["spe"]:
						return self.create_order(m)

				# checks if the opponent's pokemon doesn't have another status
				if m.id==twave.id and battle.opponent_active_pokemon.status==None:
					immune = False
					# checks if the enemy pokemon is immune
					for type in battle.opponent_active_pokemon.types:
						if type == PokemonType.ELECTRIC or type == PokemonType.GROUND:
							immune = True
					# uses thunder wave if the opponent is not immune and is faster than the active pokemon
					if not immune and estat["spe"] >= mystat["spe"]:
						return self.create_order(m)

				# checks if the opponent's pokemon doesn't have another status
				if m.id==toxic.id and battle.opponent_active_pokemon.status==None:
					immune = False
					species = battle.active_pokemon.species

					# checks if the opponent's pokemon is immune
					for type in battle.opponent_active_pokemon.types:
						if type == PokemonType.POISON or type == PokemonType.STEEL:
							immune = True

					# uses toxic if the enemy is not immune or if the active pokemon is salazzle (which has corrosion)
					if ((not immune) or (species.lower=="salazzle")):
						return self.create_order(m)

				# checks if the opponent's pokemon doesn't have another status
				if m.id==wisp.id and battle.active_pokemon.status==None:
					immune = False
					# checks if the opponent is immune
					for type in battle.opponent_active_pokemon.types:
						if type == PokemonType.FIRE:
							immune = True

					# uses wil o wisp if the enemy isn't immmune and is a physical attacker
					if not immune and estat["atk"] >= mystat["spa"]:
						return self.create_order(m)

		# switches to a new pokemon if the active pokemon fainted or has a debuffed attack or special attack stat
		if (battle.active_pokemon.fainted or battle.active_pokemon.boosts["atk"]<= -2  or battle.active_pokemon.boosts["spa"]<= -2) and len(battle.available_switches)>0:
			return switch(self,battle)

		if battle.available_moves:

			damages = [0, 0, 0, 0]

			# finds how much damage each move deals
			for i in range(len(battle.available_moves)):

				# damage calculations are found using the damageCalc helper method
				damages[i] = damageCalc(battle.active_pokemon,battle.opponent_active_pokemon, battle.available_moves[i], battle.weather, battle.fields)
			index = 0
			max = 0
			# finds the move that deals the most damage
			for i in range (len(battle.available_moves)):
				if max < damages[i]:
					max = damages[i]
					index = i

			if len(battle.available_switches) == 0 and battle._can_dynamax:
				# uses the move that deals the most damage
				return self.create_order(battle.available_moves[index],dynamax=True)

			else:
				return self.create_order(battle.available_moves[index])

		# no moves available so switch
		elif(len(battle.available_switches)!=0):
			return switch(self,battle)
		else:
			return self.choose_random_move(battle)
# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Gen8EnvSinglePlayer):
	def __init__(self, *args, **kwargs):
		Gen8EnvSinglePlayer.__init__(self)
		self.model = None

	def embed_battle(self, battle):
		# -1 indicates that the move does not have a base power
		# or is not available

		# my_pokemon = []
		# #create stuff about my pokemon
		# numpokemon_mine = 0
		# for pokemon in battle.team.values():
		# 	#embed HP
		# 	my_pokemon = np.concatenate([my_pokemon, [
		# 		pokemon.current_hp,
		# 		pokemon.current_hp_fraction,
		# 		int(pokemon.active == True),
		# 		typeToInt(pokemon.type_1),
		# 		typeToInt(pokemon.type_2),
		# 		]])
		# 	#Embed moves
		# 	movenum = 0
		# 	for name, move in pokemon.moves.items():
		# 		my_pokemon = np.concatenate([my_pokemon, [
		# 			move.accuracy,
		# 			move.base_power,
		# 			move.crit_ratio,
		# 			move.current_pp,
		# 			move.expected_hits,
		# 			typeToInt(move.type),
		# 			]])
		# 		movenum+=1
		# 	print("Move count for pokemon: " + str(movenum))
		# 	if movenum < 4:
		# 		print("Adding moves: " + str(4-movenum))
		# 		my_pokemon = np.concatenate([my_pokemon, -np.ones(6*(4-movenum))])
		# 	numpokemon_mine+=1

		# print("My pokemon count: " + str(numpokemon_mine))
		# if numpokemon_mine < 6:
		# 	print("Adding pokemon: " + str(6-numpokemon_mine))
		# 	my_pokemon = np.concatenate([my_pokemon, -np.ones(29*(6-numpokemon_mine))])

		# print("my_pokemon size: " + str(len(my_pokemon)))

		# their_pokemon = []
		# numpokemon = 0
		# for pokemon in battle.opponent_team.values():
		# 	#do something
		# 	#embed HP
		# 	their_pokemon = np.concatenate([their_pokemon, [
		# 		pokemon.current_hp,
		# 		pokemon.current_hp_fraction,
		# 		int(pokemon.active == True),
		# 		typeToInt(pokemon.type_1),
		# 		typeToInt(pokemon.type_2),
		# 		]])
		# 	#Embed moves
		# 	movenum = 0
		# 	for name, move in pokemon.moves.items():
		# 		their_pokemon = np.concatenate([their_pokemon, [
		# 			move.accuracy,
		# 			move.base_power,
		# 			move.crit_ratio,
		# 			move.current_pp,
		# 			move.expected_hits,
		# 			typeToInt(move.type),
		# 			]])
		# 		movenum+=1
		# 	print("Moves for pokemon: " + str(movenum))
		# 	if movenum < 4:
		# 		print("Adding moves: " + str(4-movenum))
		# 		their_pokemon = np.concatenate([their_pokemon, -np.ones(6*(4-movenum))])
		# 	numpokemon+=1

		# print("Number of opponent pokemon: " + str(numpokemon))
		# if numpokemon < 6:
		# 	print("Adding pokemon: " + str(6-numpokemon))
		# 	their_pokemon = np.concatenate([their_pokemon, -np.ones(29*(6-numpokemon))])

		# print("their_pokemon size: " + str(len(their_pokemon)))


		# damages = [0, 0, 0, 0]

		# for i in range(len(battle.available_moves)):
		# 		damages[i] = damageCalc(battle.active_pokemon,battle.opponent_active_pokemon, battle.available_moves[i], battle.weather, battle.fields)

		# # We count how many pokemons have not fainted in each team
		# remaining_mon_team = (
		# 	len([mon for mon in battle.team.values() if mon.fainted]) / 6
		# )
		# remaining_mon_opponent = (
		# 	len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
		# )
		# # Final vector with 354
		# returnstuff = np.concatenate(
		# 	[
		# 		damages,
		# 		my_pokemon,
		# 		their_pokemon,
		# 		[remaining_mon_team, remaining_mon_opponent]
		# 	]
		# 	)
		# if len(returnstuff) != 354:
		# 	print("SHIT BROKE")
		# 	print("SHIT BROKE")
		# 	print("SHIT BROKE")
		# 	print("SHIT BROKE")

		# return returnstuff

		damages = [0,0,0,0]
		for i in range(len(battle.available_moves)):
			damages[i] = damageCalc(battle.active_pokemon,battle.opponent_active_pokemon, battle.available_moves[i], battle.weather, battle.fields)

			

		remaining_mon_team = (
			len([mon for mon in battle.team.values() if mon.fainted]) / 6
		)
		remaining_mon_opponent = (
			len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
		)

		returnstuff = np.concatenate(
			[
				damages,
				[remaining_mon_team, remaining_mon_opponent]

			]
			)


		return returnstuff
		

	def compute_reward(self, battle) -> float:
		return self.reward_computing_helper(
			battle, fainted_value=2, hp_value=1, victory_value=30
		)



class MaxDamagePlayer(RandomPlayer):
	def choose_move(self, battle):
		# If the player can attack, it will
		if battle.available_moves:
			# Finds the best move among available ones
			best_move = max(battle.available_moves, key=lambda move: move.base_power)
			return self.create_order(best_move)

		# If no attack is available, a random switch will be made
		else:
			return self.choose_random_move(battle)


NB_TRAINING_STEPS = 10000
NB_EVALUATION_EPISODES = 100

tf.random.set_seed(0)
np.random.seed(0)

def createPlayer():
	env_player = SimpleRLPlayer(battle_format="gen8randombattle")
	# Output dimension
	n_action = len(env_player.action_space)

	model = Sequential()
	model.add(Dense(16, activation="elu", input_shape=(1,6)))
	
	model.add(Dense(64, activation="elu"))

	model.add(Dense(128, activation = "elu"))
	model.add(Dense(256, activation = "relu"))
	model.add(Dense(256, activation = "relu"))
	model.add(Dense(128, activation = "elu"))


	# Our embedding have shape (1, 10), which affects our hidden layer
	# dimension and output dimension
	# Flattening resolve potential issues that would arise otherwise
	model.add(Flatten())
	model.add(Dense(64, activation="elu"))
	model.add(Dense(16, activation="elu"))
	model.add(Dense(n_action, activation="linear"))

	memory = SequentialMemory(limit=10000, window_length=1)

	# Ssimple epsilon greedy
	policy = LinearAnnealedPolicy(
	  EpsGreedyQPolicy(),
	  attr="eps",
	  value_max=1.0,
	  value_min=0.05,
	  value_test=0,
	  nb_steps=10000,
	)

	# policy = BoltzmannQPolicy()

	# Defining our DQN
	dqn = DQNAgent(
		model=model,
		nb_actions=len(env_player.action_space),
		policy=policy,
		memory=memory,
		nb_steps_warmup=1000,
		gamma=0.5,
		target_model_update=1,
		delta_clip=0.01,
		enable_double_dqn=True,
	)

	dqn.compile(Adam(lr=1e-3), metrics=["mae"])

	env_player.model = model

	return env_player, model, dqn, policy



# This is the function that will be used to train the dqn
def dqn_training(player, dqn, nb_steps):
	dqn.fit(player, nb_steps=nb_steps)
	player.complete_current_battle()


def dqn_evaluation(player, dqn, nb_episodes):
	# Reset battle statistics
	player.reset_battles()
	dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

	print(
		"DQN Evaluation: %d victories out of %d episodes"
		% (player.n_won_battles, nb_episodes)
	)


if __name__ == "__main__":
	env_player, model, dqn, policy = createPlayer()

	rand = RandomPlayer(battle_format="gen8randombattle")
	maxp = MaxDamagePlayer(battle_format="gen8randombattle")
	huer = SimpleHeuristicsPlayer(battle_format="gen8randombattle")
	best = PokeBot(battle_format="gen8randombattle")

	for opponent in [huer, best]:
		env_player.play_against(
			env_algorithm=dqn_training,
			opponent=opponent,
			env_algorithm_kwargs={"dqn": dqn, "nb_steps": 100000},
		)
	model.save("model_%d_vsALL" % NB_TRAINING_STEPS)

	print("\nAgent Trained vs huesristic: Results against random player:")
	env_player.play_against(
		env_algorithm=dqn_evaluation,
		opponent=rand,
		env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
	)

	print("\nAgent Trained vs Huersitic: Results against max player:")
	env_player.play_against(
		env_algorithm=dqn_evaluation,
		opponent=maxp,
		env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
	)

	print("\nAgent Trained vs hueristic: Results against hueristic player:")
	env_player.play_against(
		env_algorithm=dqn_evaluation,
		opponent=huer,
		env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
	)

	print("\nAgent Trained vs hueristic: Results against best player:")
	env_player.play_against(
		env_algorithm=dqn_evaluation,
		opponent=best,
		env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
	)

