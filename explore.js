require('dotenv').config();
const { MongoClient } = require('mongodb');

const uri = process.env.MONGODB_URI;

function summarizeDoc(doc, prefix = '') {
  const fields = {};
  for (const [k, v] of Object.entries(doc)) {
    const path = prefix ? `${prefix}.${k}` : k;
    let type = Array.isArray(v) ? 'array' : v === null ? 'null' : typeof v;
    if (v && typeof v === 'object' && !Array.isArray(v) && v._bsontype) type = v._bsontype;
    fields[path] = type;
  }
  return fields;
}

(async () => {
  const client = new MongoClient(uri, { serverSelectionTimeoutMS: 15000 });
  try {
    await client.connect();
    const admin = client.db().admin();
    const { databases } = await admin.listDatabases();

    for (const dbInfo of databases) {
      if (['admin', 'local', 'config'].includes(dbInfo.name)) continue;
      const db = client.db(dbInfo.name);
      console.log(`\n=== Database: ${dbInfo.name} (${(dbInfo.sizeOnDisk / 1024 / 1024).toFixed(2)} MB) ===`);
      const collections = await db.listCollections().toArray();

      for (const collInfo of collections) {
        const coll = db.collection(collInfo.name);
        const count = await coll.countDocuments();
        console.log(`\n-- Collection: ${collInfo.name} (${count} docs) --`);
        const samples = await coll.find({}).limit(3).toArray();
        if (samples.length === 0) {
          console.log('  (empty)');
          continue;
        }
        const fieldTypes = {};
        for (const s of samples) {
          Object.assign(fieldTypes, summarizeDoc(s));
        }
        console.log('  Fields:', JSON.stringify(fieldTypes, null, 2));
        console.log('  Sample doc:', JSON.stringify(samples[0], null, 2).slice(0, 1500));
      }
    }
  } catch (err) {
    console.error('ERROR:', err.message);
  } finally {
    await client.close();
  }
})();
