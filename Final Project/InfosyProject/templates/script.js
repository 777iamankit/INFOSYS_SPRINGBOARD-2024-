function toggleMenu() {
  const menu = document.getElementById("menu");
  const closeBtn = document.getElementById("close-btn");
  const hamburger = document.getElementById("hamburger");

  menu.classList.toggle("active"); // Toggle the menu visibility
  closeBtn.classList.toggle("active"); // Toggle the visibility of the close button
  hamburger.classList.toggle("active"); // Hide hamburger icon when menu is open
}
