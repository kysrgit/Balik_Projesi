## 2026-02-25 - [Lightbox Keyboard Accessibility]
**Learning:** The custom lightbox implementation for viewing detection thumbnails was completely inaccessible to keyboard users. Once opened, it trapped the user unless they used a mouse to click the background. Modals in web applications should always support the native `Escape` key pattern.
**Action:** Always bind a global `keydown` event listener to close custom modals/lightboxes when the `Escape` key is pressed, ensuring keyboard accessibility compliance.
