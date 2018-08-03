# search@ml2lm

The search part are mainly based my other project [auto-learn-model](https://github.com/yakolle/auto-learn-model). 

But that project have some flaws, such as lack of flexibility in the learning processes, e.g. training part can't easily interact with the previous processes, the meta context infos(i.e. superparameters, evaluations, validations) isolate from the very process during transforming/encoding/selecting/training. So I will rewrite it.

The main issue of this part is hardly to organize the meta infos into one interactive context chain. The first idea is using nested E/V(Evaluation/Validation) to guide the search process, but it will take more running resources(cpus, memories, time). I will leave it for a while till I get a better idea to handle this.

To do: meta context chain
