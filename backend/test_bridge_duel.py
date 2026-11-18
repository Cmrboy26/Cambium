import unittest

import state
import game_logic
from arena.arenaClass import Arena
from BotClass import Bot
from kits.kitBase import Kit
from kits import bridgeKit


class BridgeDuelTests(unittest.TestCase):
    def setUp(self):
        state.MAP_NAME = "bridge_duel"
        state.BRIDGE_TARGET_SCORE = 3
        state.match_scores.clear()
        state.match_rounds.clear()

    def test_bridge_arena_metadata(self):
        arena = Arena(
            [[-19, 15, 25], [-36, 15, 25]],
            "open",
            "Bridge Duel 1",
            "bridge_duel",
            [-42, 14, 18, -13, 22, 32],
        )
        self.assertEqual(arena.map_name, "bridge_duel")
        self.assertEqual(arena.spawnCoords[0][0] + arena.spawnCoords[1][0], -55)

    def test_bridge_kit_has_sword_and_blocks(self):
        kit = Kit(bridgeKit.kitStart, bridgeKit.kitEnd, "Bot1")
        commands = "\n".join(kit.commands)
        self.assertIn("iron_sword", commands)
        self.assertIn("wool", commands)

    def test_target_score_resets_match(self):
        winner = Bot("Bot1", "fighting", None, "agent")
        loser = Bot("Bot2", "fighting", None, "agent")
        winner.pairAgainst(loser)
        for expected in (1, 2):
            _, score, match_won = game_logic.award_round(winner, loser)
            self.assertEqual(score, expected)
            self.assertFalse(match_won)
        key, score, match_won = game_logic.award_round(winner, loser)
        self.assertEqual(score, 3)
        self.assertTrue(match_won)
        self.assertEqual(state.match_scores[key], {"Bot1": 0, "Bot2": 0})


if __name__ == "__main__":
    unittest.main()
