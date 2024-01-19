var waterQualityResult = "{{ a }}";
var geologyResult = "{{ b }}";
var lithologyResult = "{{ c }}";

// JavaScript to open and close the feedback modal
const openFeedbackButton = document.getElementById("open-feedback");
const feedbackModal = document.getElementById("feedback-modal");
const closeButton = document.querySelector(".close");

openFeedbackButton.addEventListener("click", function () {
  feedbackModal.style.display = "block";
});

closeButton.addEventListener("click", function () {
  feedbackModal.style.display = "none";
});

window.addEventListener("click", function (event) {
  if (event.target === feedbackModal) {
    feedbackModal.style.display = "none";
  }
});

// JavaScript to handle form submission and display output
document.addEventListener('DOMContentLoaded', function () {
  const form = document.querySelector('form');
  const outputContainer = document.querySelector('.output-container');
  const closeOutputButton = document.getElementById('close-output'); // Get the close button

  form.addEventListener('submit', async function (e) {
      e.preventDefault();

      // Get user input
      const user_input = document.getElementById('user_input').value;

      // Perform form submission and get the results using AJAX
      const response = await fetch('/process_input', {
          method: 'POST',
          body: new URLSearchParams({ user_input }),
          headers: {
              'Content-Type': 'application/x-www-form-urlencoded'
          }
      });

      if (response.ok) {
          const results = await response.json();

          // Display the results in the output container
          const outputContent = document.querySelector('.output-content');
          outputContent.innerHTML = `
              <h2>Results:</h2>
              <p>Water Quality: ${results.water_quality_result}</p>
              <p>Geology: ${results.geology_result}</p>
              <p>Lithology: ${results.lithology_result}</p>
          `;

          // Show the output container
          outputContainer.style.right = '0';
      }
  });

  // Add an event listener to close the output container when the "OK" button is clicked
  closeOutputButton.addEventListener('click', function () {
      outputContainer.style.right = '-100%'; // Hide the output container by moving it to the right
  });
});

// JavaScript for Star Rating
const starRatingInputs = document.querySelectorAll('input[name="rating"]');
let selectedRating = null; // Variable to store the selected rating

starRatingInputs.forEach((input) => {
    input.addEventListener('change', function () {
        selectedRating = this.value; // Store the selected rating
    });
});