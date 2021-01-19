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
import asyncio


# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Player):
    model = None

    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

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
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
                [int(battle.trapped)]
            ]
        )

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
    )

    


    await huerPlayer.ladder(5)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())



