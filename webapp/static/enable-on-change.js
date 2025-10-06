(function(){
    // Delegated handler: listen at document level so dynamically-added inputs are caught.
    function handleEvent(e){
        try{
            var target = e.target || e.srcElement;
            if(!target) return;
            // only consider input/select/textarea events
            var tag = (target.tagName || '').toLowerCase();
            if(['input','select','textarea'].indexOf(tag) === -1) return;

            // Find the nearest form (if any)
            var form = target.closest('form');

            // Determine candidate buttons to enable: those with data-enable-on-change="true"
            var candidates = [];
            if(form){
                candidates = Array.prototype.slice.call(form.querySelectorAll('button[data-enable-on-change="true"]'));
            } else {
                candidates = Array.prototype.slice.call(document.querySelectorAll('button[data-enable-on-change="true"]'));
            }

            candidates.forEach(function(btn){
                if(btn.disabled){
                    console.debug('[enable-on-change] enabling button', btn.id || btn.name || btn);
                    btn.disabled = false;
                    // restore styling if inline disabled style present
                    try{ btn.style.cursor = ''; btn.style.background = ''; }catch(e){}
                }
            });
        }catch(err){
            console.warn('[enable-on-change] handler error', err);
        }
    }

    if(typeof document !== 'undefined'){
        // Use capture to catch events early in case other handlers stop propagation
        document.addEventListener('input', handleEvent, true);
        document.addEventListener('change', handleEvent, true);

        // Also run once on DOMContentLoaded to log status
        function initLog(){
            try{
                var buttons = document.querySelectorAll('button[data-enable-on-change="true"]');
                if(buttons.length){
                    console.debug('[enable-on-change] found', buttons.length, 'buttons to monitor');
                }
            }catch(e){}
        }
        if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initLog);
        else initLog();
    }
})();
