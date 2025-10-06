(function(){
    // Find buttons with data-enable-on-change="true" and enable them when any input/select/textarea
    // within the same form is changed or receives input.
    function initEnableOnChange(){
        var buttons = Array.prototype.slice.call(document.querySelectorAll('button[data-enable-on-change="true"]'));
        buttons.forEach(function(btn){
            // if not inside a form, observe the whole document
            var form = btn.closest('form');
            if(!form){
                // watch document-level inputs
                var inputs = Array.prototype.slice.call(document.querySelectorAll('input, select, textarea'));
            } else {
                var inputs = Array.prototype.slice.call(form.querySelectorAll('input, select, textarea'));
            }

            if(!btn.disabled) return; // already enabled

            var enableBtn = function(){
                btn.disabled = false;
                // optionally update styling if inline styles exist
                try{ btn.style.cursor = ''; btn.style.background = ''; } catch(e){}
                inputs.forEach(function(i){ i.removeEventListener('change', enableBtn); i.removeEventListener('input', enableBtn); });
            };

            inputs.forEach(function(i){ i.addEventListener('change', enableBtn); i.addEventListener('input', enableBtn); });
        });
    }

    if(document.readyState === 'loading'){
        document.addEventListener('DOMContentLoaded', initEnableOnChange);
    } else {
        initEnableOnChange();
    }
})();
