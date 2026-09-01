import json
from evaluation.adversarial_suite import AdversarialRedTeamSuite

suite = AdversarialRedTeamSuite()
res = suite.run_full_suite()
print(json.dumps(res, indent=2))
