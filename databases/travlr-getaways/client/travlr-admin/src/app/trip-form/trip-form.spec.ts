import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TripForm } from './trip-form';

describe('TripForm', () => {
  let component: TripForm;
  let fixture: ComponentFixture<TripForm>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TripForm]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TripForm);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
