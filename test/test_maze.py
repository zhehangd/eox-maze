import unittest

import numpy as np

import maze

class TestMaze(unittest.TestCase):
    
    def setUp(self):
        self.maze = maze.MazeCore()
        
    def test_is_wall(self):
        self.assertEqual(self.maze.is_wall((0,0)), True)
        self.assertEqual(self.maze.is_wall((1,0)), True)
        self.assertEqual(self.maze.is_wall((0,1)), True)
        self.assertEqual(self.maze.is_wall((1,1)), False)
        self.assertEqual(self.maze.is_wall((1,4)), True)
        self.assertEqual(self.maze.is_wall((4,1)), False)
        
    def test_is_player(self):
        player_pos = self.maze.player_pos
        another_pos = (player_pos[0]+1,player_pos[1])
        self.assertEqual(self.maze.is_player(player_pos), True)
        self.assertEqual(self.maze.is_player(another_pos), False)
    
    def test_is_goal(self):
        goal_pos = self.maze.goal_pos
        another_pos = (goal_pos[0]+1,goal_pos[1])
        self.assertEqual(self.maze.is_goal(goal_pos), True)
        self.assertEqual(self.maze.is_goal(another_pos), False)
        
    def test_find_empty_pos(self):
        is_empty = lambda pos : not self.maze.is_wall(pos)
        pos_list = [self.maze.find_empty_pos(is_empty) for i in range(100)]
        self.assertGreater(len(set(pos_list)), 1) # very likely
        for pos in pos_list:
            self.assertEqual(self.maze.is_wall(pos), False)
    
    def test_set_player_pos(self):
        is_empty = lambda pos : not self.maze.is_wall(pos)
        pos_list = [self.maze.find_empty_pos(is_empty) for i in range(100)]
        for pos in pos_list:
            self.maze.set_player_pos(pos)
            self.assertEqual(pos, self.maze.player_pos)
    
    def test_generate_observation_1(self):
        self.maze.set_player_pos((21, 29))
        self.maze.goal_pos = (21, 29)
        vision = self.maze.generate_observation((21, 30))
        ans = np.zeros((5,5),np.float)
        ans -= np.array([[1,0,0,1,0],
                         [0,0,1,1,0],
                         [0,0,0,1,0],
                         [1,0,1,1,0],
                         [0,0,0,1,0]],np.float)
        ans[2,1] = 1
        ans = ans.reshape(-1)
        self.assertEqual(np.all(np.equal(ans, vision)), True)
    
    def test_generate_observation_2(self):
        self.maze.set_player_pos((21, 29))
        vision = self.maze.generate_observation((1, 0))
        ans = np.zeros((5,5),np.float)
        ans -= np.array([[0,0,0,0,0],
                         [0,0,1,1,1],
                         [0,0,1,0,0],
                         [0,0,1,0,1],
                         [0,0,1,0,1]],np.float)
        ans = ans.reshape(-1)
        self.assertEqual(np.all(np.equal(ans, vision)), True)

    def test_step(self):
        self.maze.goal_pos = (21, 18)
        self.maze.set_player_pos((21,21))
        vision, reward, is_end, _ = self.maze.step(0)
        self.assertEqual(self.maze.player_pos, (21,21))
        self.assertEqual(is_end, False)
        vision, reward, is_end, _ = self.maze.step(3)
        self.assertEqual(self.maze.player_pos, (20,21))
        self.assertEqual(is_end, False)
        vision, reward, is_end, _ = self.maze.step(2)
        self.assertEqual(self.maze.player_pos, (20,20))
        self.assertEqual(is_end, False)
        vision, reward, is_end, _ = self.maze.step(2)
        self.assertEqual(self.maze.player_pos, (20,19))
        self.assertEqual(is_end, False)
        vision, reward, is_end, _ = self.maze.step(1)
        self.assertEqual(self.maze.player_pos, (21,19))
        self.assertEqual(is_end, False)
        vision, reward, is_end, _ = self.maze.step(1)
        self.assertEqual(self.maze.player_pos, (21,19))
        self.assertEqual(is_end, False)
        vision, reward, is_end, _ = self.maze.step(2)
        self.assertEqual(self.maze.player_pos, (21,18))
        self.assertEqual(is_end, True)
    
if __name__ == '__main__':
    unittest.main() 
