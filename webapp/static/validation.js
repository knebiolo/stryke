/**
 * Input Validation System for Stryke Web App
 * Provides real-time validation with helpful error messages
 */

(function() {
    'use strict';

    // Validation rules configuration
    const validationRules = {
        // Number validations
        positive: {
            test: (value) => parseFloat(value) > 0,
            message: 'Must be a positive number'
        },
        nonNegative: {
            test: (value) => parseFloat(value) >= 0,
            message: 'Must be zero or greater'
        },
        percentage: {
            test: (value) => {
                const num = parseFloat(value);
                return !isNaN(num) && num >= 0 && num <= 100;
            },
            message: 'Must be between 0 and 100'
        },
        probability: {
            test: (value) => {
                const num = parseFloat(value);
                return !isNaN(num) && num >= 0 && num <= 1;
            },
            message: 'Must be between 0 and 1'
        },
        integer: {
            test: (value) => Number.isInteger(parseFloat(value)),
            message: 'Must be a whole number'
        },
        positiveInteger: {
            test: (value) => {
                const num = parseFloat(value);
                return Number.isInteger(num) && num > 0;
            },
            message: 'Must be a positive whole number'
        },
        
        // Range validations
        range: {
            test: (value, min, max) => {
                const num = parseFloat(value);
                return !isNaN(num) && num >= min && num <= max;
            },
            message: (min, max) => `Must be between ${min} and ${max}`
        },
        
        // Required field
        required: {
            test: (value) => value !== null && value !== undefined && value.toString().trim() !== '',
            message: 'This field is required'
        },
        
        // Text validations
        minLength: {
            test: (value, length) => value.length >= length,
            message: (length) => `Must be at least ${length} characters`
        },
        maxLength: {
            test: (value, length) => value.length <= length,
            message: (length) => `Must be no more than ${length} characters`
        },
        
        // Date validations
        validDate: {
            test: (value) => !isNaN(Date.parse(value)),
            message: 'Must be a valid date'
        },
        
        // Email validation
        email: {
            test: (value) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value),
            message: 'Must be a valid email address'
        }
    };

    // Custom validation for sum of percentages/probabilities
    function validatePercentageSum(inputs, targetSum = 100, tolerance = 0.01) {
        const values = inputs.map(input => parseFloat(input.value) || 0);
        const sum = values.reduce((a, b) => a + b, 0);
        return Math.abs(sum - targetSum) <= tolerance;
    }

    // Custom validation for flow values
    function validateFlowRange(value) {
        const num = parseFloat(value);
        if (isNaN(num)) return { valid: false, message: 'Must be a number' };
        if (num < 0) return { valid: false, message: 'Flow cannot be negative' };
        if (num > 1000000) return { valid: false, message: 'Flow seems unreasonably high' };
        return { valid: true };
    }

    // Custom validation for turbine parameters
    function validateTurbineParams(rpm, diameter, blades) {
        const errors = [];
        
        if (rpm && (rpm < 1 || rpm > 1000)) {
            errors.push('RPM should be between 1 and 1000');
        }
        
        if (diameter && (diameter < 0.1 || diameter > 50)) {
            errors.push('Diameter should be between 0.1 and 50 meters');
        }
        
        if (blades && (blades < 1 || blades > 20)) {
            errors.push('Number of blades should be between 1 and 20');
        }
        
        return errors;
    }

    // Add validation to an input element
    function addValidation(input, rules) {
        if (!input) return;
        
        // Store validation rules on the element
        input.validationRules = rules;
        
        // Create error message container
        let errorDiv = input.nextElementSibling;
        if (!errorDiv || !errorDiv.classList.contains('validation-error')) {
            errorDiv = document.createElement('div');
            errorDiv.className = 'validation-error';
            errorDiv.style.cssText = `
                color: #dc3545;
                font-size: 13px;
                margin-top: 4px;
                display: none;
                font-weight: normal;
            `;
            input.parentNode.insertBefore(errorDiv, input.nextSibling);
        }
        
        // Add event listeners
        input.addEventListener('blur', () => validateInput(input));
        input.addEventListener('input', () => {
            // Clear error on input if field becomes valid
            if (input.classList.contains('invalid')) {
                setTimeout(() => validateInput(input), 300);
            }
        });
        
        // Initial validation if has value
        if (input.value) {
            validateInput(input);
        }
    }

    // Validate a single input
    function validateInput(input, silent = false) {
        const rules = input.validationRules;
        if (!rules) return true;
        
        const value = input.value;
        const errorDiv = input.nextElementSibling;
        let isValid = true;
        let errorMessage = '';
        
        // Check each rule
        for (const rule of rules) {
            const ruleConfig = validationRules[rule.type];
            if (!ruleConfig) continue;
            
            let testResult;
            if (rule.params) {
                testResult = ruleConfig.test(value, ...rule.params);
            } else {
                testResult = ruleConfig.test(value);
            }
            
            if (!testResult) {
                isValid = false;
                if (typeof ruleConfig.message === 'function') {
                    errorMessage = ruleConfig.message(...(rule.params || []));
                } else {
                    errorMessage = ruleConfig.message;
                }
                break;
            }
        }
        
        // Update UI
        if (!silent) {
            if (isValid) {
                input.classList.remove('invalid');
                input.classList.add('valid');
                if (errorDiv && errorDiv.classList.contains('validation-error')) {
                    errorDiv.style.display = 'none';
                }
            } else {
                input.classList.add('invalid');
                input.classList.remove('valid');
                if (errorDiv && errorDiv.classList.contains('validation-error')) {
                    errorDiv.textContent = '⚠️ ' + errorMessage;
                    errorDiv.style.display = 'block';
                }
            }
        }
        
        return isValid;
    }

    // Validate entire form
    function validateForm(form) {
        const inputs = form.querySelectorAll('input[data-validate], select[data-validate], textarea[data-validate]');
        let isValid = true;
        let firstInvalidInput = null;
        
        inputs.forEach(input => {
            if (!validateInput(input, false)) {
                isValid = false;
                if (!firstInvalidInput) {
                    firstInvalidInput = input;
                }
            }
        });
        
        // Scroll to first invalid input
        if (!isValid && firstInvalidInput) {
            firstInvalidInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
            firstInvalidInput.focus();
        }
        
        return isValid;
    }

    // Validate percentage sum (for operating scenarios, population distributions, etc.)
    function validatePercentageGroup(groupName, targetSum = 100) {
        const inputs = document.querySelectorAll(`[data-percentage-group="${groupName}"]`);
        if (inputs.length === 0) return true;
        
        const sum = Array.from(inputs).reduce((acc, input) => {
            return acc + (parseFloat(input.value) || 0);
        }, 0);
        
        const isValid = Math.abs(sum - targetSum) < 0.01;
        const errorDiv = document.getElementById(`${groupName}-sum-error`);
        
        if (!isValid) {
            inputs.forEach(input => input.classList.add('invalid'));
            if (errorDiv) {
                errorDiv.textContent = `⚠️ Total must equal ${targetSum}% (currently ${sum.toFixed(1)}%)`;
                errorDiv.style.display = 'block';
            }
        } else {
            inputs.forEach(input => input.classList.remove('invalid'));
            if (errorDiv) {
                errorDiv.style.display = 'none';
            }
        }
        
        return isValid;
    }

    // Initialize validation on page load
    function initializeValidation() {
        // Add CSS for validation states
        const style = document.createElement('style');
        style.textContent = `
            input.invalid, select.invalid, textarea.invalid {
                border-color: #dc3545 !important;
                background-color: #fff5f5 !important;
            }
            
            input.valid, select.valid, textarea.valid {
                border-color: #28a745 !important;
            }
            
            .validation-error {
                animation: shake 0.3s ease-in-out;
            }
            
            @keyframes shake {
                0%, 100% { transform: translateX(0); }
                25% { transform: translateX(-5px); }
                75% { transform: translateX(5px); }
            }
            
            .validation-summary {
                background: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 4px;
                padding: 12px;
                margin: 15px 0;
                display: none;
            }
            
            .validation-summary.show {
                display: block;
            }
            
            .validation-summary h4 {
                margin: 0 0 8px 0;
                color: #856404;
                font-size: 16px;
            }
            
            .validation-summary ul {
                margin: 0;
                padding-left: 20px;
                color: #856404;
            }
        `;
        document.head.appendChild(style);
        
        // Parse data-validate attributes and set up validation
        document.querySelectorAll('[data-validate]').forEach(input => {
            const validateAttr = input.getAttribute('data-validate');
            const rules = parseValidationRules(validateAttr);
            addValidation(input, rules);
        });
        
        // Intercept form submissions
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', function(e) {
                // Skip validation for certain forms (file uploads, templates)
                if (form.classList.contains('no-validate')) {
                    return true;
                }
                
                if (!validateForm(form)) {
                    e.preventDefault();
                    showValidationSummary(form);
                    return false;
                }
            });
        });
        
        // Set up percentage group validation
        document.querySelectorAll('[data-percentage-group]').forEach(input => {
            const groupName = input.getAttribute('data-percentage-group');
            input.addEventListener('input', () => {
                setTimeout(() => validatePercentageGroup(groupName), 300);
            });
        });
    }

    // Parse validation rules from data-validate attribute
    function parseValidationRules(validateStr) {
        const rules = [];
        const parts = validateStr.split('|');
        
        parts.forEach(part => {
            const [type, ...params] = part.split(':');
            const rule = { type: type.trim() };
            
            if (params.length > 0) {
                rule.params = params[0].split(',').map(p => {
                    const num = parseFloat(p);
                    return isNaN(num) ? p.trim() : num;
                });
            }
            
            rules.push(rule);
        });
        
        return rules;
    }

    // Show validation summary
    function showValidationSummary(form) {
        let summary = form.querySelector('.validation-summary');
        if (!summary) {
            summary = document.createElement('div');
            summary.className = 'validation-summary';
            form.insertBefore(summary, form.firstChild);
        }
        
        const invalidInputs = form.querySelectorAll('.invalid');
        if (invalidInputs.length === 0) {
            summary.classList.remove('show');
            return;
        }
        
        const errors = Array.from(invalidInputs).map(input => {
            const label = form.querySelector(`label[for="${input.id}"]`);
            const fieldName = label ? label.textContent : input.name || 'Field';
            const errorDiv = input.nextElementSibling;
            const errorMsg = errorDiv && errorDiv.classList.contains('validation-error') 
                ? errorDiv.textContent.replace('⚠️ ', '') 
                : 'Invalid value';
            return `${fieldName}: ${errorMsg}`;
        });
        
        summary.innerHTML = `
            <h4>⚠️ Please fix the following errors:</h4>
            <ul>
                ${errors.map(err => `<li>${err}</li>`).join('')}
            </ul>
        `;
        summary.classList.add('show');
        summary.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // Public API
    window.StrykeValidation = {
        validate: validateInput,
        validateForm: validateForm,
        validatePercentageGroup: validatePercentageGroup,
        addValidation: addValidation,
        rules: validationRules
    };

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeValidation);
    } else {
        initializeValidation();
    }
})();
