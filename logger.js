console.log(__filename);
console.log(__dirname);

var url='http://mylogger.io/log';

function log(message)
{
    console.log('i am function');
}

module.exports.log =log;
module.exports.url=url;   //module.exports.endpoint also we can change name which is visible to outside

module.exports=log;       //in module we can export single function or an object

/////////////////////////////module wrapper/////////////////////////////////////////////

(function (exports,require,module,__filename,__dirname)
{
    console.log(__filename);
    console.log(__dirname);

    var url='http://mylogger.io/log';

function log(message)
{
    console.log('i am function');
}

module.exports=log;       //in module we can export single function or an object

})
//////////////////////////////////////////////////////////////

const EventEmitter=require('events');
var url='http://mylogger.io/log';
class Logger extends EventEmitter{
    log(message)
    {
        //send an http request
        console.log(message);

        //raise an event
        this.emit('messagelogged',{id:1, url: 'http://'});
    }
}
module.exports= Logger;