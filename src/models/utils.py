from audiocraft.modules.codebooks_patterns import CodebooksPatternProvider, Pattern, PatternLayout, LayoutCoord
import typing as tp
class EmptyPatternProvider(CodebooksPatternProvider):
    
    def __init__(self, n_q: int, delays: tp.Optional[tp.List[int]] = None, empty_initial: int = 0):
        super().__init__(n_q)
        if delays is None:
            delays = list(range(n_q))
        self.delays = delays
        self.empty_initial = empty_initial

    def get_pattern(self, timesteps: int) -> Pattern:
        out: PatternLayout = [[]]
        max_delay = max(self.delays)
        if self.empty_initial:
            out += [[] for _ in range(self.empty_initial)]
        for t in range(0, timesteps):
            v = []
            for q, delay in enumerate(self.delays):
                    v.append(LayoutCoord(t, q))
            out.append(v)

        
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)

