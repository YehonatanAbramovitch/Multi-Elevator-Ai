// Check pySetOrder (src/core.js) against real CPython frozenset orders.
// Usage: python -c "import json,random; random.seed(1); c=[sorted(random.sample(range(40),random.randint(1,12))) for _ in range(3000)]; json.dump([[x,list(frozenset(tuple(x)))] for x in c],open('sets.json','w'))"
//        node verify/pyset_check.js sets.json
const fs = require('fs');
const path = require('path');
const { pySetOrder } = require(path.join(__dirname, '..', 'src', 'core.js'));
const cases = JSON.parse(fs.readFileSync(process.argv[2] || 'sets.json', 'utf8'));
const bad = cases.filter(([items, order]) => pySetOrder(items).join() !== order.join());
console.log(`sets=${cases.length} mismatches=${bad.length}`);
process.exit(bad.length ? 1 : 0);
