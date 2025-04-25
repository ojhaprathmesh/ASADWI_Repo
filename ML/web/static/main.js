// AquaSense - Main JavaScript File
document.addEventListener('DOMContentLoaded', () => {
    // Initialize elements
    initializeMobileMenu();
    initializeForms();
});

// Mobile Menu Toggle
function initializeMobileMenu() {
    const menuToggle = document.querySelector('.menu-toggle');
    const navMenu = document.querySelector('.nav-menu');

    if (menuToggle && navMenu) {
        menuToggle.addEventListener('click', () => {
            navMenu.classList.toggle('active');
            menuToggle.classList.toggle('active');
        });
    }

    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (navMenu && navMenu.classList.contains('active') &&
            !e.target.closest('.nav-menu') &&
            !e.target.closest('.menu-toggle')) {
            navMenu.classList.remove('active');
            menuToggle.classList.remove('active');
        }
    });
}

// Initialize Form Handlers
function initializeForms() {
    // Water Quality Form
    const waterQualityForm = document.getElementById('water-quality-form');
    if (waterQualityForm) {
        waterQualityForm.addEventListener('submit', handlePredictionSubmit);
    }

    // Geological Form
    const geologicalForm = document.getElementById('geological-form');
    if (geologicalForm) {
        geologicalForm.addEventListener('submit', handlePredictionSubmit);
    }

    // Lithology Form
    const lithologyForm = document.getElementById('lithology-form');
    if (lithologyForm) {
        lithologyForm.addEventListener('submit', handlePredictionSubmit);
    }

    // Contact Form
    const contactForm = document.getElementById('contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', handleContactSubmit);
    }

    // Reset buttons
    const resetButtons = document.querySelectorAll('.reset-form');
    resetButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            const form = e.target.closest('form');
            if (form) {
                form.reset();
                // Hide any results that might be showing
                const resultsContainer = document.getElementById('results-container');
                if (resultsContainer) {
                    showPlaceholder();
                }
            }
        });
    });

    // Prediction type selector
    const predictionTypeSelect = document.getElementById('prediction-type');
    if (predictionTypeSelect) {
        predictionTypeSelect.addEventListener('change', handlePredictionTypeChange);
        // Initialize forms visibility based on initial selection
        handlePredictionTypeChange.call(predictionTypeSelect);
    }
}

// Handle prediction type change
function handlePredictionTypeChange() {
    const selectedValue = this.value;

    // Get all forms
    const waterQualityForm = document.getElementById('water-quality-form');
    const geologicalForm = document.getElementById('geological-form');
    const lithologyForm = document.getElementById('lithology-form');

    // Hide all forms
    if (waterQualityForm) waterQualityForm.classList.add('hidden');
    if (geologicalForm) geologicalForm.classList.add('hidden');
    if (lithologyForm) lithologyForm.classList.add('hidden');

    // Show the selected form
    if (selectedValue === 'water-quality' && waterQualityForm) {
        waterQualityForm.classList.remove('hidden');
    } else if (selectedValue === 'geological' && geologicalForm) {
        geologicalForm.classList.remove('hidden');
    } else if (selectedValue === 'lithology' && lithologyForm) {
        lithologyForm.classList.remove('hidden');
    }

    // Reset the results container
    showPlaceholder();
}

// Handle prediction form submission
function handlePredictionSubmit(e) {
    e.preventDefault();

    const form = e.target;
    const formType = form.getAttribute('id');
    const formData = new FormData(form);

    // Show loading state
    showLoading();

    // Determine the endpoint based on form type
    let endpoint = '';
    if (formType === 'water-quality-form') {
        endpoint = '/predict/water_quality';
    } else if (formType === 'geological-form') {
        endpoint = '/geology';
    } else if (formType === 'lithology-form') {
        endpoint = '/lithology';
    }

    // Make API request
    fetch(endpoint, {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Check if there's an error in the response
            if (data.error) {
                throw new Error(data.error);
            }
            // Display results
            displayResults(data, formType);
        })
        .catch(error => {
            console.error('Error:', error);
            showError('An error occurred while processing your request. Please try again.');
        });
}

// Handle contact form submission
function handleContactSubmit(e) {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);
    const submitButton = form.querySelector('button[type="submit"]');

    // Disable button during submission
    if (submitButton) {
        submitButton.disabled = true;
        submitButton.textContent = 'Sending...';
    }

    // Send form data
    fetch('/contact', {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Show success message
            const formMessage = document.createElement('div');
            formMessage.className = 'form-message success';
            formMessage.textContent = 'Thank you for your message! We will get back to you soon.';

            // Insert message after form
            form.parentNode.insertBefore(formMessage, form.nextSibling);

            // Reset form
            form.reset();

            // Remove message after a delay
            setTimeout(() => {
                formMessage.remove();
            }, 5000);
        })
        .catch(error => {
            console.error('Error:', error);

            // Show error message
            const formMessage = document.createElement('div');
            formMessage.className = 'form-message error';
            formMessage.textContent = 'An error occurred. Please try again later.';

            // Insert message after form
            form.parentNode.insertBefore(formMessage, form.nextSibling);

            // Remove message after a delay
            setTimeout(() => {
                formMessage.remove();
            }, 5000);
        })
        .finally(() => {
            // Re-enable button
            if (submitButton) {
                submitButton.disabled = false;
                submitButton.textContent = 'Send Message';
            }
        });
}

// Display prediction results
function displayResults(data, formType) {
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) return;

    // Clear previous content
    resultsContainer.innerHTML = '';

    // Create result content
    const resultDiv = document.createElement('div');
    resultDiv.className = 'prediction-result';

    // Add result header
    const header = document.createElement('h3');
    let title = '';
    switch (formType) {
        case 'water-quality-form':
            title = 'Water Quality Prediction';
            break;
        case 'geological-form':
            title = 'Geological Prediction';
            break;
        case 'lithology-form':
            title = 'Lithology Prediction';
            break;
        default:
            title = 'Prediction Result';
    }
    header.textContent = title;
    resultDiv.appendChild(header);

    // Add result value
    const resultValue = document.createElement('div');
    resultValue.className = 'result-value';

    // Format the result based on prediction type
    if (data.result !== undefined) {
        resultValue.textContent = data.result;

        // Add color coding for water quality results
        if (formType === 'water-quality-form') {
            if (data.result.includes('Safe')) {
                resultValue.style.color = 'var(--success-color, green)';
            } else {
                resultValue.style.color = 'var(--danger-color, red)';
            }
        }
    } else if (data.status) {
        // For older API format
        resultValue.textContent = data.status;
        if (data.status.includes('Safe')) {
            resultValue.style.color = 'var(--success-color, green)';
        } else {
            resultValue.style.color = 'var(--danger-color, red)';
        }
    } else {
        resultValue.textContent = 'No result available';
    }
    resultDiv.appendChild(resultValue);

    // Add details if available
    if (data.details) {
        const detailsDiv = document.createElement('div');
        detailsDiv.className = 'result-details';

        const detailsHeader = document.createElement('h4');
        detailsHeader.textContent = 'Details';
        detailsDiv.appendChild(detailsHeader);

        const detailsList = document.createElement('ul');
        for (const [key, value] of Object.entries(data.details)) {
            const listItem = document.createElement('li');
            listItem.textContent = `${key}: ${value}`;
            detailsList.appendChild(listItem);
        }
        detailsDiv.appendChild(detailsList);
        resultDiv.appendChild(detailsDiv);
    }

    // Add visualization if available
    if (data.chart || data.chart_url) {
        const chartDiv = document.createElement('div');
        chartDiv.className = 'result-chart';

        const chartImg = document.createElement('img');
        chartImg.src = data.chart || data.chart_url;
        chartImg.alt = 'Prediction Visualization';
        chartImg.className = 'result-chart-image';

        chartDiv.appendChild(chartImg);
        resultDiv.appendChild(chartDiv);
    }

    // Append to results container
    resultsContainer.appendChild(resultDiv);

    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

// Show loading state
function showLoading() {
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) return;

    resultsContainer.innerHTML = `
        <div class="placeholder-content">
            <div class="spinner"></div>
            <p>Processing your request...</p>
        </div>
    `;
}

// Show error message
function showError(message) {
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) return;

    resultsContainer.innerHTML = `
        <div class="placeholder-content">
            <div class="icon-large">❌</div>
            <p>${message}</p>
        </div>
    `;
}

// Show placeholder content
function showPlaceholder() {
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) return;

    resultsContainer.innerHTML = `
        <div class="placeholder-content">
            <div class="icon-large">🔍</div>
            <p>Select a prediction type and fill out the form to see results.</p>
        </div>
    `;
} 