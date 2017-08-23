// Shorthand for $( document ).ready()
$(function() {
    $("table").addClass('table table-striped table-hover');
    $(".highlight-ipython3").addClass('zoom');
    $(".section ol p").addClass('zoom');
    $(".section ul p").addClass('zoom');
    $("li > a").click(function() {
      $( this ).children( "span.toctree-expand").click();
    });
});
