pub fn rate(p1: &Rating, p2: &Rating, outcome: &Outcome, cfg: &SkillrCfg) -> (Rating, Rating) {
    let d_loc = p1.loc - p2.loc;
    let s_diff = (p1.scale.powi(2) + p2.scale.powi(2) + 2.0 * cfg.beta.powi(2)).sqrt();
    let [p_win, p_draw, p_lose] = _probs(d_loc, s_diff, cfg);
    let new_p1 = p1.update(s_diff, [p_win, p_draw, p_lose], outcome, cfg);
    let new_p2 = p2.update(s_diff, [p_lose, p_draw, p_win], &outcome.inv(), cfg);
    (new_p1, new_p2)
}

/// From the perspective of player 1, the probabilities of winning, drawning, losing
pub fn probs(p1: &Rating, p2: &Rating, cfg: &SkillrCfg) -> [f64; 3] {
    let d_loc = p1.loc - p2.loc;
    let s_diff = (p1.scale.powi(2) + p2.scale.powi(2) + 2.0 * cfg.beta.powi(2)).sqrt();
    _probs(d_loc, s_diff, cfg)
}

fn _probs(d_loc: f64, s_diff: f64, cfg: &SkillrCfg) -> [f64; 3] {
    let z = fd(cfg.p_draw, d_loc, s_diff);
    let p_win = 1.0 - z;
    let p_lose = fd(-cfg.p_draw, d_loc, s_diff);
    let p_draw = z - p_lose;
    [p_win, p_draw, p_lose]
}

fn fd(x: f64, d_loc: f64, s_diff: f64) -> f64 {
    sigma((x - d_loc) / s_diff)
}

fn sigma(z: f64) -> f64 {
    (1.0 + (-z).exp()).recip()
}

pub struct SkillrCfg {
    pub loc: f64,
    pub scale: f64,
    pub beta: f64,
    pub tau: f64,
    pub p_draw: f64,
    pub entropy_rate: f64,
}

#[derive(Debug, Clone)]
pub struct Rating {
    pub loc: f64,
    pub scale: f64,
}

#[derive(Clone)]
pub enum Outcome {
    Win,
    Draw,
    Loss,
}

impl Rating {
    pub fn new(loc: f64, scale: f64) -> Self {
        Self { loc, scale }
    }

    fn update(&self, s_diff: f64, ps: [f64; 3], outcome: &Outcome, cfg: &SkillrCfg) -> Self {
        let (p, o) = match outcome {
            Outcome::Win => (ps[0], 1.0),
            Outcome::Draw => (ps[1], 0.0),
            Outcome::Loss => (ps[2], -1.0),
        };
        let expectation = ps[0] - ps[2];
        let information_gain = -p.ln();
        let k = self.scale.powi(2) / s_diff.powi(2);
        let new_loc = self.update_loc(k, o, expectation);
        let new_scale = self.update_scale(information_gain, cfg);
        Self {
            loc: new_loc,
            scale: (new_scale.powi(2) + cfg.tau.powi(2)).sqrt(),
        }
    }

    fn update_loc(&self, k: f64, o: f64, expectation: f64) -> f64 {
        self.loc + k * (o - expectation)
    }

    fn update_scale(&self, information_gain: f64, cfg: &SkillrCfg) -> f64 {
        let entropy = self.scale.ln() + 2.0;
        let new_entropy = entropy - cfg.entropy_rate * information_gain;
        (new_entropy - 2.0).exp()
    }
}

impl SkillrCfg {
    pub fn rating(&self) -> Rating {
        Rating::new(self.loc, self.scale)
    }
}

impl Outcome {
    pub fn inv(&self) -> Self {
        match self {
            Self::Win => Self::Loss,
            Self::Draw => Self::Draw,
            Self::Loss => Self::Win,
        }
    }
}

impl From<&Outcome> for i8 {
    fn from(value: &Outcome) -> Self {
        match value {
            Outcome::Win => 1,
            Outcome::Draw => 0,
            Outcome::Loss => -1,
        }
    }
}

impl From<&Outcome> for String {
    fn from(value: &Outcome) -> Self {
        match value {
            Outcome::Win => "Win".to_owned(),
            Outcome::Draw => "Draw".to_owned(),
            Outcome::Loss => "Loss".to_owned(),
        }
    }
}

impl Default for Rating {
    fn default() -> Self {
        Self::new(25.0, 25.0 / 3.0)
    }
}

impl Default for SkillrCfg {
    fn default() -> Self {
        SkillrCfg {
            loc: 25.0,
            scale: 25.0 / 3.0,
            beta: 25.0 / 6.0,
            tau: 25.0 / 300.0,
            p_draw: 0.1,
            entropy_rate: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Outcome, Rating, SkillrCfg, rate};

    #[test]
    fn a() {
        let p1 = Rating::default();
        let p2 = Rating::default();
        let cfg = SkillrCfg::default();
        let (new_p1, new_p2) = rate(&p1, &p2, &Outcome::Win, &cfg);

        assert!(new_p1.loc > p1.loc);
        assert!(new_p2.loc < p2.loc);
        assert!(new_p1.scale < p1.scale);
        assert!(new_p2.scale < p2.scale);
    }
}
