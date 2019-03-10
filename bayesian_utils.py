import pandas as pd
import pymc3 as pm, theano.tensor as tt
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np

def data_cleaner(path = ''):
    df = pd.read_csv(path)
    df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
    
    df = df.rename(index = str, columns={'HomeTeam': 'home_team', 'AwayTeam': 'away_team', 'FTHG':'home_score', 'FTAG':'away_score'})
    
    return df
    
def set_model(num_teams, home_team, away_team, observed_home_goals, observed_away_goals):
    
    with pm.Model() as model:
    # global model parameters
        home = pm.Flat('home')
        sd_att = pm.HalfStudentT('sd_att', nu=3, sd=2.5)
        sd_def = pm.HalfStudentT('sd_def', nu=3, sd=2.5)
        intercept = pm.Flat('intercept')

        # team-specific model parameters
        atts_star = pm.Normal("atts_star", mu=0, sd=sd_att, shape=num_teams)
        defs_star = pm.Normal("defs_star", mu=0, sd=sd_def, shape=num_teams)

        atts = pm.Deterministic('atts', atts_star - tt.mean(atts_star))
        defs = pm.Deterministic('defs', defs_star - tt.mean(defs_star))
        home_theta = tt.exp(intercept + home + atts[home_team] + defs[away_team])
        away_theta = tt.exp(intercept + atts[away_team] + defs[home_team])

        # likelihood of observed data
        home_points = pm.Poisson('home_points', mu=home_theta, observed=observed_home_goals)
        away_points = pm.Poisson('away_points', mu=away_theta, observed=observed_away_goals)
        
    return model
        
def train_model(model, num_samples, discard = 2000, cores = 4):
    with model:
        trace = pm.sample(num_samples, tune = discard, cores = cores)
    
    return trace

def team_parameter_explorer(trace, teams, param = ''):
    
    df_hpd = pd.DataFrame(pm.stats.hpd(trace[param]),
                      columns=['hpd_low', 'hpd_high'],
                      index=teams.team.values)
    df_median = pd.DataFrame(pm.stats.quantiles(trace[param])[50],
                         columns=['hpd_median'],
                         index=teams.team.values)
    df_hpd = df_hpd.join(df_median)
    df_hpd['relative_lower'] = df_hpd.hpd_median - df_hpd.hpd_low
    df_hpd['relative_upper'] = df_hpd.hpd_high - df_hpd.hpd_median
    df_hpd = df_hpd.sort_values(by='hpd_median')
    df_hpd = df_hpd.reset_index()
    df_hpd['x'] = df_hpd.index + .5


    fig, axs = plt.subplots(figsize=(14,4))
    axs.errorbar(df_hpd.x, df_hpd.hpd_median,
             yerr=(df_hpd[['relative_lower', 'relative_upper']].values).T,
             fmt='o')
    axs.set_title(f'HPD of {param} Strength by Team')
    axs.set_xlabel('Team')
    axs.set_ylabel('Posterior Attack Strength')
    axs.set_xticks(df_hpd.index + .5)
    axs.set_xticklabels(df_hpd['index'].values, rotation=45)
    
    return axs

def match_score(trace, df, home_team, away_team, sample_size=1000):
    
    team_1 = df[df['home_team'] == home_team].i_home.unique()[0]
    team_2 = df[df['away_team'] == away_team].i_away.unique()[0]
    
    samples = np.random.randint(0, 
                                trace['intercept'].shape[0], 
                                size=sample_size)
    
    intercept_ = trace['intercept'][samples]
    
    offs_ = trace['atts'][samples]
    defs_ = trace['defs'][samples]
    
    home_theta_ = np.exp(intercept_ + offs_[:,  team_1] + defs_[:, team_2])
    away_theta_ = np.exp(intercept_ + offs_[:,  team_2] + defs_[:,  team_1])
    
    home_score_ = np.random.poisson(home_theta_, sample_size)
    away_score_ = np.random.poisson(away_theta_, sample_size)
    
    home_win_prob = np.mean((home_score_ - away_score_ > 0))
    draw_prob = np.mean((home_score_ == away_score_))
    away_win_prob = np.mean((home_score_ - away_score_ < 0))
    
    if home_win_prob > away_win_prob and home_win_prob > draw_prob:
        winner = home_team
    elif home_win_prob < away_win_prob and away_win_prob > draw_prob:
        winner = away_team
    else:
        winner = 'draw'
        
    return {
        'winner': winner,
        'probs': {
            'home_team': home_win_prob,
            'away_team': away_win_prob,
            'draw': draw_prob
        },
        'home_team_goals': {
            '95%': np.percentile(home_score_, 5),
            '75%': np.percentile(home_score_, 25),
            '50%': np.percentile(home_score_, 50),
            '25%': np.percentile(home_score_, 70),
            '10%': np.percentile(home_score_, 90),
            '05%': np.percentile(home_score_, 95)
        },
        'away_team_goals': {
            '95%': np.percentile(away_score_, 5),
            '75%': np.percentile(away_score_, 25),
            '50%': np.percentile(away_score_, 50),
            '25%': np.percentile(away_score_, 70),
            '10%': np.percentile(away_score_, 90),
            '05%': np.percentile(away_score_, 95)
        }
    }