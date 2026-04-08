const fs = require('fs');
const buf = fs.readFileSync('eyedetecter.tflite');
console.log('Magic:', buf.toString('utf8', 4, 8));
