import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import random
from collections import deque
from datetime import datetime
import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import tkinter as tk
from tkinter import ttk, colorchooser
import re

class ConfigGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Tag Game Configuration")
        self.root.geometry("400x600")
        
        style = ttk.Style()
        style.configure('TButton', padding=5)
        style.configure('TLabel', padding=5)
        
        self.config = {}
        self._create_widgets()
        
    def _create_widgets(self):
        # Add resolution settings at the top
        ttk.Label(self.root, text="Window Resolution:").pack()
        resolution_frame = ttk.Frame(self.root)
        resolution_frame.pack(pady=5)
        
        ttk.Label(resolution_frame, text="Width:").pack(side=tk.LEFT)
        self.window_width = ttk.Entry(resolution_frame, width=8)
        self.window_width.insert(0, "1024")
        self.window_width.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(resolution_frame, text="Height:").pack(side=tk.LEFT)
        self.window_height = ttk.Entry(resolution_frame, width=8)
        self.window_height.insert(0, "768")
        self.window_height.pack(side=tk.LEFT, padx=5)

        # Grid Size
        ttk.Label(self.root, text="Grid Size:").pack()
        self.grid_size = ttk.Entry(self.root)
        self.grid_size.insert(0, "50")
        self.grid_size.pack()
        
        # Colors
        ttk.Label(self.root, text="Colors:").pack()
        color_frame = ttk.Frame(self.root)
        color_frame.pack(pady=5)
        
        self.tagger_color = "#FF0000"
        self.runner_color = "#0000FF"
        
        ttk.Button(color_frame, text="Choose Tagger Color", 
                  command=lambda: self._choose_color('tagger')).pack(side=tk.LEFT, padx=5)
        ttk.Button(color_frame, text="Choose Runner Color", 
                  command=lambda: self._choose_color('runner')).pack(side=tk.LEFT, padx=5)
        
        # Speed Settings
        ttk.Label(self.root, text="Tagger Speed:").pack()
        self.tagger_speed = ttk.Entry(self.root)
        self.tagger_speed.insert(0, "2")
        self.tagger_speed.pack()
        
        # Training Settings
        ttk.Label(self.root, text="Number of Episodes:").pack()
        self.num_episodes = ttk.Entry(self.root)
        self.num_episodes.insert(0, "1000")
        self.num_episodes.pack()
        
        ttk.Label(self.root, text="Visualization Delay (ms):").pack()
        self.viz_delay = ttk.Entry(self.root)
        self.viz_delay.insert(0, "50")
        self.viz_delay.pack()
        
        # Obstacle Settings
        ttk.Label(self.root, text="Number of Random Obstacles:").pack()
        self.num_obstacles = ttk.Entry(self.root)
        self.num_obstacles.insert(0, "4")
        self.num_obstacles.pack()
        
        # Trail Settings
        self.show_trails = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.root, text="Show Movement Trails", 
                       variable=self.show_trails).pack(pady=5)
        
        # Start Button
        ttk.Button(self.root, text="Start Game", 
                  command=self._start_game).pack(pady=20)
        
    def _choose_color(self, agent_type):
        color = colorchooser.askcolor(title=f"Choose {agent_type} color")[1]
        if color:
            if agent_type == 'tagger':
                self.tagger_color = color
            else:
                self.runner_color = color
    
    def _validate_hex_color(self, color):
        return bool(re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color))
    
    def _start_game(self):
        try:
            self.config = {
                'window_width': int(self.window_width.get()),
                'window_height': int(self.window_height.get()),
                'grid_size': int(self.grid_size.get()),
                'tagger_speed': int(self.tagger_speed.get()),
                'tagger_color': self.tagger_color,
                'runner_color': self.runner_color,
                'num_episodes': int(self.num_episodes.get()),
                'viz_delay': int(self.viz_delay.get()),
                'show_trails': self.show_trails.get(),
                'num_obstacles': int(self.num_obstacles.get()),
                'max_steps': 200,
                'batch_size': 64,
                'learning_rate': 0.001,
                'gamma': 0.99,
                'initial_epsilon': 1.0,
                'min_epsilon': 0.01,
                'epsilon_decay': 0.995,
                'viz_interval': 1,
                'save_interval': 100
            }
            
            self.root.destroy()
        except ValueError as e:
            tk.messagebox.showerror("Error", "Please enter valid numerical values")

class TagEnvironment:
    def __init__(self, grid_size=100, tagger_speed=2, num_obstacles=4):
        self.grid_size = grid_size
        self.tagger_speed = tagger_speed
        self.num_obstacles = num_obstacles
        self.obstacles = self._generate_random_obstacles()
        self.reset()
        self.tagger_trail = []
        self.runner_trail = []
        
    def _generate_random_obstacles(self):
        obstacles = []
        for _ in range(self.num_obstacles):
            while True:
                pos = np.array([random.randint(0, self.grid_size-1),
                              random.randint(0, self.grid_size-1)])
                if not any(np.array_equal(pos, obs) for obs in obstacles):
                    obstacles.append(pos)
                    break
        return obstacles
    
    def reset(self):
        # Reset trails
        self.tagger_trail = []
        self.runner_trail = []
        
        # Place agents away from obstacles and each other
        while True:
            self.tagger_pos = np.array([random.randint(0, self.grid_size//4),
                                      random.randint(0, self.grid_size//4)])
            if not self._is_obstacle(self.tagger_pos):
                break
        
        while True:
            self.runner_pos = np.array([random.randint(3*self.grid_size//4, self.grid_size-1),
                                      random.randint(3*self.grid_size//4, self.grid_size-1)])
            if not self._is_obstacle(self.runner_pos) and \
               not np.array_equal(self.runner_pos, self.tagger_pos):
                break
        
        # Initialize trails with starting positions
        self.tagger_trail.append(self.tagger_pos.copy())
        self.runner_trail.append(self.runner_pos.copy())
        
        return self._get_state()
    
    def _is_obstacle(self, pos):
        return any(np.array_equal(pos, obs) for obs in self.obstacles)
    
    def step(self, tagger_action, runner_action):
        # Update trails
        self.tagger_trail.append(self.tagger_pos.copy())
        self.runner_trail.append(self.runner_pos.copy())
        
        # Keep only last 10 positions for trail
        if len(self.tagger_trail) > 10:
            self.tagger_trail.pop(0)
        if len(self.runner_trail) > 10:
            self.runner_trail.pop(0)
        
        # Tagger moves multiple times based on speed advantage
        for _ in range(self.tagger_speed):
            new_pos = self.tagger_pos.copy()
            self._move(new_pos, tagger_action)
            if not self._is_obstacle(new_pos):
                self.tagger_pos = new_pos

        # Runner moves once
        new_pos = self.runner_pos.copy()
        self._move(new_pos, runner_action)
        if not self._is_obstacle(new_pos):
            self.runner_pos = new_pos

        # Calculate distance-based reward
        distance = np.linalg.norm(self.tagger_pos - self.runner_pos)
        done = np.array_equal(self.tagger_pos, self.runner_pos)
        
        # Enhanced reward structure
        if done:
            reward = 10  # Tagger catches runner
        else:
            # Reward includes distance component and obstacle avoidance
            reward = -0.1 - (distance / self.grid_size)
            # Small bonus for staying away from obstacles
            if not any(np.linalg.norm(self.tagger_pos - obs) < 2 for obs in self.obstacles):
                reward += 0.05
            
        return self._get_state(), reward, done
    
    def _move(self, position, action):
        if action == 0: position[1] = max(0, position[1] - 1)  # Up
        elif action == 1: position[1] = min(self.grid_size - 1, position[1] + 1)  # Down
        elif action == 2: position[0] = max(0, position[0] - 1)  # Left
        elif action == 3: position[0] = min(self.grid_size - 1, position[0] + 1)  # Right
    
    def _get_state(self):
        return np.concatenate([self.tagger_pos, self.runner_pos])

class GameVisualizer:
    def __init__(self, grid_size, window_width, window_height, tagger_color="#FF0000", 
                 runner_color="#0000FF", show_trails=True):
        self.grid_size = grid_size
        self.window_width = window_width
        self.window_height = window_height
        self.stats_height = 200  # Height reserved for stats panel
        self.show_trails = show_trails  # Store show_trails as instance variable
        
        # Calculate cell size based on window dimensions and grid size
        self.cell_size = min(
            (window_width) // grid_size,
            (window_height - self.stats_height) // grid_size
        )
        
        # Calculate actual game area dimensions
        self.game_width = self.cell_size * grid_size
        self.game_height = self.cell_size * grid_size
        
        # Calculate padding to center the game area
        self.padding_x = (window_width - self.game_width) // 2
        self.padding_y = (window_height - self.stats_height - self.game_height) // 2
        
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Enhanced RL Tag Game")
        
        # Enhanced colors with transparency
        self.colors = {
            'background': (240, 240, 245),  # Slight blue tint
            'grid': (220, 220, 230),        # Subtle grid lines
            'tagger': self._hex_to_rgb(tagger_color),
            'runner': self._hex_to_rgb(runner_color),
            'obstacle': (100, 100, 120, 180),  # Semi-transparent obstacles
            'text': (60, 60, 80),             # Dark blue-grey text
            'stats_bg': (250, 250, 255),      # Light background for stats
            'trail_tagger': (*self._hex_to_rgb(tagger_color), 40),  # More transparent trails
            'trail_runner': (*self._hex_to_rgb(runner_color), 40)
        }
        
        # Initialize fonts with better sizing
        self.title_font = pygame.font.Font(None, 36)
        self.stats_font = pygame.font.Font(None, 28)
        self.info_font = pygame.font.Font(None, 24)
    
    def _hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _draw_rounded_rect(self, surface, color, rect, radius=10):
        """Draw a rounded rectangle"""
        pygame.draw.rect(surface, color, rect, border_radius=radius)
    
    def _draw_agent(self, pos, color, is_tagger=False):
        """Draw an agent with a more sophisticated appearance"""
        x = self.padding_x + pos[0] * self.cell_size
        y = self.padding_y + pos[1] * self.cell_size
        size = self.cell_size - 2  # Slightly smaller than cell
        
        # Draw shadow
        shadow_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(shadow_surface, (0, 0, 0, 30), (size//2, size//2), size//2)
        self.screen.blit(shadow_surface, (x + 2, y + 2))
        
        # Draw agent body
        agent_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(agent_surface, (*color, 255), (size//2, size//2), size//2)
        
        # Add highlight
        highlight_size = size // 3
        pygame.draw.circle(agent_surface, (255, 255, 255, 100),
                         (size//3, size//3), highlight_size)
        
        self.screen.blit(agent_surface, (x, y))
        
        # Add indicator for tagger
        if is_tagger:
            indicator_size = size // 4
            pygame.draw.circle(self.screen, (255, 255, 255),
                             (x + size//2, y + size//2), indicator_size, 2)
    
    def draw(self, env, stats, episode, epsilon):
        # Fill background
        self.screen.fill(self.colors['background'])
        
        # Draw game area background
        game_rect = pygame.Rect(self.padding_x, self.padding_y,
                              self.game_width, self.game_height)
        pygame.draw.rect(self.screen, self.colors['stats_bg'], game_rect)
        
        # Draw grid
        for i in range(self.grid_size + 1):
            x = self.padding_x + i * self.cell_size
            y = self.padding_y + i * self.cell_size
            
            pygame.draw.line(self.screen, self.colors['grid'],
                           (x, self.padding_y),
                           (x, self.padding_y + self.game_height))
            pygame.draw.line(self.screen, self.colors['grid'],
                           (self.padding_x, y),
                           (self.padding_x + self.game_width, y))
        
        # Draw trails
        if self.show_trails:
            for pos in env.tagger_trail[:-1]:
                x = self.padding_x + pos[0] * self.cell_size
                y = self.padding_y + pos[1] * self.cell_size
                trail_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                pygame.draw.circle(trail_surface, self.colors['trail_tagger'],
                                 (self.cell_size//2, self.cell_size//2),
                                 self.cell_size//3)
                self.screen.blit(trail_surface, (x, y))
            
            for pos in env.runner_trail[:-1]:
                x = self.padding_x + pos[0] * self.cell_size
                y = self.padding_y + pos[1] * self.cell_size
                trail_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                pygame.draw.circle(trail_surface, self.colors['trail_runner'],
                                 (self.cell_size//2, self.cell_size//2),
                                 self.cell_size//3)
                self.screen.blit(trail_surface, (x, y))
        
        # Draw obstacles with shadows
        for obs in env.obstacles:
            x = self.padding_x + obs[0] * self.cell_size
            y = self.padding_y + obs[1] * self.cell_size
            size = self.cell_size - 2
            
            # Draw shadow
            shadow_surface = pygame.Surface((size, size), pygame.SRCALPHA)
            self._draw_rounded_rect(shadow_surface, (0, 0, 0, 50),
                                  pygame.Rect(2, 2, size-4, size-4), 5)
            self.screen.blit(shadow_surface, (x+2, y+2))
            
            # Draw obstacle
            obstacle_surface = pygame.Surface((size, size), pygame.SRCALPHA)
            self._draw_rounded_rect(obstacle_surface, self.colors['obstacle'],
                                  pygame.Rect(0, 0, size-4, size-4), 5)
            self.screen.blit(obstacle_surface, (x, y))
        
        # Draw agents
        self._draw_agent(env.tagger_pos, self.colors['tagger'], True)
        self._draw_agent(env.runner_pos, self.colors['runner'], False)
        
        # Draw stats panel
        stats_rect = pygame.Rect(0, self.window_height - self.stats_height,
                               self.window_width, self.stats_height)
        pygame.draw.rect(self.screen, self.colors['stats_bg'], stats_rect)
        pygame.draw.line(self.screen, self.colors['grid'],
                        (0, stats_rect.top), (self.window_width, stats_rect.top), 2)
        
        # Draw stats with enhanced layout
        title = self.title_font.render("Training Statistics", True, self.colors['text'])
        self.screen.blit(title, (20, stats_rect.top + 10))
        
        stats_data = [
            (f"Episode: {episode}", f"Epsilon: {epsilon:.3f}"),
            (f"Tagger Wins: {stats.tagger_wins}", f"Runner Wins: {stats.runner_wins}"),
            (f"Win Rate: {(stats.tagger_wins/(stats.tagger_wins + stats.runner_wins)*100):.1f}%" 
             if stats.tagger_wins + stats.runner_wins > 0 else "Win Rate: N/A",
             f"Avg Reward: {np.mean(stats.episode_rewards[-100:]):.2f}"),
            (f"Avg Episode Length: {np.mean(stats.episode_lengths[-100:]):.1f}",
             f"Distance: {np.linalg.norm(env.tagger_pos - env.runner_pos):.1f}")
        ]
        
        for i, (left_text, right_text) in enumerate(stats_data):
            y_pos = stats_rect.top + 50 + i * 30
            left_surface = self.stats_font.render(left_text, True, self.colors['text'])
            right_surface = self.stats_font.render(right_text, True, self.colors['text'])
            self.screen.blit(left_surface, (20, y_pos))
            self.screen.blit(right_surface, (self.window_width // 2 + 20, y_pos))
        
        pygame.display.flip()

def main():
    # Create and show configuration GUI
    config_gui = ConfigGUI()
    config_gui.root.mainloop()
    
    # Get configuration from GUI
    config = config_gui.config
    if not config:  # If window was closed without starting
        return
    
    # Create environment with custom settings
    env = TagEnvironment(grid_size=config['grid_size'],
                        tagger_speed=config['tagger_speed'],
                        num_obstacles=config['num_obstacles'])
    
    # Initialize agents and training components
    tagger_agent = DQN(4, 4)
    runner_agent = DQN(4, 4)
    tagger_optimizer = optim.Adam(tagger_agent.parameters(), lr=config['learning_rate'])
    runner_optimizer = optim.Adam(runner_agent.parameters(), lr=config['learning_rate'])
    
    tagger_buffer = ExperienceBuffer()
    runner_buffer = ExperienceBuffer()

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ExperienceBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class StatsTracker:
    def __init__(self):
        self.episode_rewards = []
        self.tagger_wins = 0
        self.runner_wins = 0
        self.episode_lengths = []
        self.avg_distances = []
        self.moving_avg_reward = []
        self.win_rates = []
        
    def update(self, reward, length, avg_distance, tagger_won):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.avg_distances.append(avg_distance)
        
        if tagger_won:
            self.tagger_wins += 1
        else:
            self.runner_wins += 1
            
        # Calculate moving averages
        window = min(100, len(self.episode_rewards))
        if window > 0:
            self.moving_avg_reward.append(np.mean(self.episode_rewards[-window:]))
            total_games = self.tagger_wins + self.runner_wins
            self.win_rates.append(self.tagger_wins / total_games if total_games > 0 else 0)

def train_agents(config):
    env = TagEnvironment(grid_size=config['grid_size'],
                        tagger_speed=config['tagger_speed'],
                        num_obstacles=config['num_obstacles'])
    
    tagger_agent = DQN(4, 4)
    runner_agent = DQN(4, 4)
    tagger_optimizer = optim.Adam(tagger_agent.parameters(), lr=config['learning_rate'])
    runner_optimizer = optim.Adam(runner_agent.parameters(), lr=config['learning_rate'])
    
    tagger_buffer = ExperienceBuffer()
    runner_buffer = ExperienceBuffer()
    stats = StatsTracker()
    
    visualizer = GameVisualizer(
        config['grid_size'],
        window_width=config['window_width'],
        window_height=config['window_height'],
        tagger_color=config['tagger_color'],
        runner_color=config['runner_color'],
        show_trails=config['show_trails']
    )  
    
    epsilon = config['initial_epsilon']
    best_performance = float('-inf')
    
    for episode in range(config['num_episodes']):
        state = env.reset()
        total_reward = 0
        steps = 0
        distances = []
        
        while steps < config['max_steps']:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return stats
                
            # Select actions
            tagger_action = select_action(tagger_agent, state, epsilon)
            runner_action = select_action(runner_agent, state, epsilon)
            
            # Take step
            next_state, reward, done = env.step(tagger_action, runner_action)
            
            # Store experiences
            tagger_buffer.push((state, tagger_action, reward, next_state, done))
            runner_buffer.push((state, runner_action, -reward, next_state, done))
            
            # Train agents if enough experience is collected
            if len(tagger_buffer) > config['batch_size']:
                train_step(tagger_agent, tagger_optimizer, tagger_buffer, config)
                train_step(runner_agent, runner_optimizer, runner_buffer, config)
            
            total_reward += reward
            distances.append(np.linalg.norm(env.tagger_pos - env.runner_pos))
            steps += 1
            
            # Visualization
            if episode % config['viz_interval'] == 0:
                visualizer.draw(env, stats, episode, epsilon)
                pygame.time.wait(config['viz_delay'])
            
            if done:
                break
            
            state = next_state
        
        # Update statistics
        stats.update(total_reward, steps, np.mean(distances), done)
        
        # Save best model
        current_performance = np.mean(stats.episode_rewards[-100:])
        if current_performance > best_performance:
            best_performance = current_performance
            save_checkpoint(episode, tagger_agent, runner_agent, stats, config, is_best=True)
        
        # Regular checkpoint saving
        if episode % config['save_interval'] == 0:
            save_checkpoint(episode, tagger_agent, runner_agent, stats, config)
        
        # Decay epsilon
        epsilon = max(config['min_epsilon'],
                     epsilon * config['epsilon_decay'])
        
        # Plot training progress periodically
        if episode % 100 == 0:
            plot_training_progress(stats)
    
    return stats

def select_action(model, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        with torch.no_grad():
            return torch.argmax(model(torch.FloatTensor(state))).item()

def train_step(model, optimizer, buffer, config):
    batch = buffer.sample(config['batch_size'])
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    
    current_q = model(states).gather(1, actions.unsqueeze(1))
    next_q = model(next_states).max(1)[0].detach()
    target_q = rewards + config['gamma'] * next_q * (1 - dones)
    
    loss = nn.MSELoss()(current_q.squeeze(), target_q)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def save_checkpoint(episode, tagger_agent, runner_agent, stats, config, is_best=False):
    checkpoint = {
        'episode': episode,
        'tagger_state_dict': tagger_agent.state_dict(),
        'runner_state_dict': runner_agent.state_dict(),
        'stats': {
            'episode_rewards': stats.episode_rewards,
            'tagger_wins': stats.tagger_wins,
            'runner_wins': stats.runner_wins,
            'episode_lengths': stats.episode_lengths,
            'avg_distances': stats.avg_distances,
            'moving_avg_reward': stats.moving_avg_reward,
            'win_rates': stats.win_rates
        },
        'config': config
    }
    
    os.makedirs('checkpoints', exist_ok=True)
    if is_best:
        torch.save(checkpoint, 'checkpoints/best_model.pt')
    else:
        torch.save(checkpoint, f'checkpoints/checkpoint_{episode}.pt')

def plot_training_progress(stats):
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(stats.moving_avg_reward)
    plt.title('Moving Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot win rates
    plt.subplot(2, 2, 2)
    plt.plot(stats.win_rates)
    plt.title('Tagger Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    
    # Plot episode lengths
    plt.subplot(2, 2, 3)
    plt.plot(stats.episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # Plot average distances
    plt.subplot(2, 2, 4)
    plt.plot(stats.avg_distances)
    plt.title('Average Tagger-Runner Distance')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/training_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

def main():
    # Create and show configuration GUI
    config_gui = ConfigGUI()
    config_gui.root.mainloop()
    
    # Get configuration from GUI
    config = config_gui.config
    if not config:  # If window was closed without starting
        return
    
    # Train agents
    stats = train_agents(config)
    
    # Save final results
    with open("results.json", "w") as f:
        json.dump({
            'episode_rewards': stats.episode_rewards,
            'tagger_wins': stats.tagger_wins,
            'runner_wins': stats.runner_wins,
            'episode_lengths': stats.episode_lengths,
            'avg_distances': stats.avg_distances,
            'moving_avg_reward': stats.moving_avg_reward,
            'win_rates': stats.win_rates
        }, f)
    
    pygame.quit()

if __name__ == "__main__":
    main()